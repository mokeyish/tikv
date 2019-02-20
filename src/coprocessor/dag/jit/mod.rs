// Copyright 2019 PingCAP, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// See the License for the specific language governing permissions and
// limitations under the License.


#![deny(
//missing_docs,
trivial_numeric_casts,
unused_extern_crates,
unstable_features
)]
#![warn(unused_import_braces)]
#![cfg_attr(feature = "clippy", plugin(clippy(conf_file = "../../clippy.toml")))]
#![cfg_attr(feature = "cargo-clippy", allow(clippy::new_without_default))]
#![cfg_attr(
feature = "cargo-clippy",
warn(
clippy::float_arithmetic,
clippy::mut_mut,
clippy::nonminimal_bool,
clippy::option_map_unwrap_or,
clippy::option_map_unwrap_or_else,
clippy::print_stdout,
clippy::unicode_not_nfc,
clippy::use_self
)
)]
#![allow(unused_imports)]
#![allow(dead_code)]


mod backend;
mod memory;
mod scalar_func;

pub use self::backend::{SimpleJITBackend, SimpleJITBuilder};

pub use crate::coprocessor::codec::{Error, Result as JitResult};
use crate::coprocessor::codec::Datum;
use crate::coprocessor::dag::expr::{EvalContext, Column};
use crate::util::codec::number;
use tipb::expression::Expr;
use tipb::expression::ExprType;
use tipb::expression::ScalarFuncSig;
use tipb::expression::FieldType;


use cranelift::prelude::*;
use cranelift_module::DataContext;
use cranelift_module::Module;
use cranelift_module::FuncId;
use cranelift_module::DataId;
use cranelift_module::Backend;
use cranelift_module::FuncOrDataId;
use cranelift_module::Linkage;
use cranelift_codegen::Context;
use std::mem;
use std::result::Result::Ok;
use std::collections::HashMap;


#[inline]
fn i64_to_datum(x: i64) -> Box<Datum> {
    Box::new(Datum::I64(x))
}

#[inline]
fn f64_to_datum(x: f64) -> Box<Datum>  {
    Box::new(Datum::F64(x))
}

#[inline]
fn u64_to_datum(x: u64) -> Box<Datum>  {
    Box::new(Datum::U64(x))
}

struct Jit {
    /// The function builder context, which is reused across multiple
    /// FunctionBuilder instances.
    func_ctx: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separate this for `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: Context,

    /// The data context, which is to data objects what `ctx` is to functions.
    data_ctx: DataContext,

    /// The module, with the simplejit backend, which manages the jit's functions
    module: Module<SimpleJITBackend>,
}

impl Jit {
    pub fn new() -> Jit {
        let mut module = Module::new({
            let mut builder = SimpleJITBuilder::new();
            builder.symbol("f64_pow_f64", scalar_func::f64_pow_f64 as *const u8);
            builder.symbol("i64_abs", scalar_func::i64_abs as *const u8);
            builder.symbol("f64_abs", scalar_func::f64_abs as *const u8);

            builder.symbol("i64_to_datum", i64_to_datum as *const u8);
            builder.symbol("f64_to_datum", f64_to_datum as *const u8);

            builder
        });
        let int_ptr = module.target_config().pointer_type();

        let ctx = module.make_context();
        let func_ctx = FunctionBuilderContext::new();
        let data_ctx = DataContext::new();


        let mut f64_abs_sig = module.make_signature();
        f64_abs_sig.params.push(AbiParam::new(types::F64));
        f64_abs_sig.returns.push(AbiParam::new(types::F64));

        module.declare_function("f64_abs", Linkage::Import, &f64_abs_sig)
            .unwrap();

        let mut i64_abs_sig = module.make_signature();
        i64_abs_sig.params.push(AbiParam::new(types::I64));
        i64_abs_sig.returns.push(AbiParam::new(types::I64));

        module.declare_function("i64_abs", Linkage::Import, &i64_abs_sig)
            .unwrap();


        let mut f64_pow_f64_sig = module.make_signature();
        f64_pow_f64_sig.params.push(AbiParam::new(types::F64));
        f64_pow_f64_sig.params.push(AbiParam::new(types::F64));
        f64_pow_f64_sig.returns.push(AbiParam::new(types::F64));

        module.declare_function("f64_pow_f64", Linkage::Import, &f64_pow_f64_sig)
            .unwrap();


        let mut i64_to_datum_sig = module.make_signature();
        i64_to_datum_sig.params.push(AbiParam::new(types::I64));
        i64_to_datum_sig.returns.push(AbiParam::new(int_ptr));

        module.declare_function("i64_to_datum", Linkage::Import, &i64_to_datum_sig)
            .unwrap();

        let mut f64_to_datum_sig = module.make_signature();
        f64_to_datum_sig.params.push(AbiParam::new(types::F64));
        f64_to_datum_sig.returns.push(AbiParam::new(int_ptr));

        module.declare_function("f64_to_datum", Linkage::Import, &f64_to_datum_sig)
            .unwrap();

        Jit {
            module,
            ctx,
            func_ctx,
            data_ctx,
        }
    }
    pub fn build(&mut self, _ctx: &EvalContext, expr: Expr) -> JitResult<fn(&[Datum]) -> Box<Datum>>{
        debug!(
            "build-expr";
            "expr" => ?expr
        );

        let func_id = self.build_func(expr)?;

        self.module.finalize_definitions();
        let func_ptr = self.module.get_finalized_function(func_id);
        Ok(unsafe { mem::transmute::<_, fn(&[Datum]) -> Box<Datum>>(func_ptr) })
    }


    //noinspection RsTypeCheck
    fn build_func(&mut self, mut expr: Expr) -> JitResult<FuncId> {

        let _field_type = expr.take_field_type();

        let int_ptr = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(int_ptr));
        sig.returns.push(AbiParam::new(int_ptr));

        let func = self.module
            .declare_function("expr1666", Linkage::Local, &sig)
            .unwrap();

        self.ctx.func.signature = sig;
        self.ctx.func.name = ExternalName::user(0, func.as_u32());
        {
            let mut bcx: FunctionBuilder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let ebb = bcx.create_ebb();
            bcx.switch_to_block(ebb);

            bcx.append_ebb_params_for_function_params(ebb);
            let row = bcx.ebb_params(ebb)[0];

            let mut trans = FunctionTranslator {
                builder: bcx,
                row,
                module: &mut self.module,
            };


            let value = match trans.translate_expr( expr)? {
                TypedValue::I64(v) => {
                    let func_id = trans.get_func_id("i64_to_datum").unwrap();

                    let i64_to_datum_func = trans.module.declare_func_in_func(func_id, &mut trans.builder.func);

                    let call = trans.builder.ins().call(i64_to_datum_func, &[v]);

                    let results = trans.builder.inst_results(call);
                    assert_eq!(results.len(), 1);
                    results[0].clone()
                },
                TypedValue::F64(v) => {

                    let func_id = trans.get_func_id("f64_to_datum").unwrap();

                    let f64_to_datum_func = trans.module.declare_func_in_func(func_id, &mut trans.builder.func);

                    let call = trans.builder.ins().call(f64_to_datum_func, &[v]);

                    let results = trans.builder.inst_results(call);
                    assert_eq!(results.len(), 1);
                    results[0].clone()
                },
                _ => unimplemented!()
            };

            trans.builder.ins().return_(&[value]);

            trans.builder.seal_all_blocks();
            trans.builder.finalize();
        }

        self.module.define_function(func, &mut self.ctx).unwrap();

        let _p = format!("{:?}", self.ctx.func);

        self.module.clear_context(&mut self.ctx);

        Ok(func)
    }


    pub fn build1(&mut self, _ctx: &EvalContext, expr: Expr) -> JitResult<fn(&[Datum]) -> f64>{
        debug!(
            "build-expr";
            "expr" => ?expr
        );

        let func_id = self.build_func1(expr)?;

        self.module.finalize_definitions();
        let func_ptr = self.module.get_finalized_function(func_id);
        Ok(unsafe { mem::transmute::<_, fn(&[Datum]) -> f64>(func_ptr) })
    }

    //noinspection RsTypeCheck
    fn build_func1(&mut self, mut expr: Expr) -> JitResult<FuncId> {

        let _field_type = expr.take_field_type();

        let int_ptr = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(int_ptr));
        sig.returns.push(AbiParam::new(types::F64));

        let func = self.module
            .declare_function("expr1666", Linkage::Export, &sig)
            .unwrap();

        self.ctx.func.signature = sig;
        self.ctx.func.name = ExternalName::user(0, func.as_u32());
        {
            let mut bcx: FunctionBuilder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let ebb = bcx.create_ebb();
            bcx.switch_to_block(ebb);

            bcx.append_ebb_params_for_function_params(ebb);
            let row = bcx.ebb_params(ebb)[0];

            let mut trans = FunctionTranslator {
                builder: bcx,
                row,
                module: &mut self.module,
            };

            match trans.translate_expr(expr)? {
                TypedValue::F64(v) => {
                    trans.builder.ins().return_(&[v]);
                },
                _ => unimplemented!()
            }
            trans.builder.seal_all_blocks();
            trans.builder.finalize();
        }

        self.module.define_function(func, &mut self.ctx).unwrap();

        let _p = format!("{:?}", self.ctx.func);
        self.module.clear_context(&mut self.ctx);

        Ok(func)
    }

}

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum TypedValue {
    Column(Value),
    I64(Value),
    F64(Value)
}

impl TypedValue {
    fn value(&self) -> Value {
        match *self {
            TypedValue::Column(v) => v,
            TypedValue::I64(v) => v,
            TypedValue::F64(v) => v,
        }
    }
}

struct FunctionTranslator<'a> {
    builder: FunctionBuilder<'a>,
    module: &'a mut Module<SimpleJITBackend>,
    row: Value,
}

impl<'a> FunctionTranslator<'a> {
    //noinspection RsTypeCheck
    fn translate_expr(&mut self, mut expr: Expr) -> JitResult<TypedValue> {
        match expr.get_tp() {
            ExprType::Int64 => {
                let v = number::decode_i64(&mut expr.get_val())?;
                Ok(TypedValue::I64(self.builder.ins().iconst(types::I64, v)))
            },
            ExprType::Float32 | ExprType::Float64  => {
                let v = number::decode_f64(&mut expr.get_val())?;
                Ok(TypedValue::F64(self.builder.ins().f64const(Ieee64::with_float(v))))
            },
            ExprType::ColumnRef => {
                unimplemented!()
            }
            ExprType::ScalarFunc => {
                expr.take_children()
                    .into_iter()
                    .map(|child| self.translate_expr(child))
                    .collect::<JitResult<Vec<_>>>()
                    .map(|children: Vec<TypedValue>| {
                        match expr.get_sig() {
                            ScalarFuncSig::AbsInt | ScalarFuncSig::AbsReal => {
                                assert_eq!(children.len(), 1);
                                match children[0] {
                                    TypedValue::I64(v) => {
                                        let func_id = self.get_func_id("i64_abs").unwrap();
                                        let local_func = self.module.declare_func_in_func(func_id, &mut self.builder.func);

                                        let call = self.builder.ins().call(local_func, &[v]);

                                        let results = self.builder.inst_results(call);
                                        assert_eq!(results.len(), 1);
                                        TypedValue::I64(results[0].clone())
                                    },
                                    TypedValue::F64(v) => {
                                        let func_id = self.get_func_id("f64_abs").unwrap();
                                        let local_func = self.module.declare_func_in_func(func_id, &mut self.builder.func);

                                        let call = self.builder.ins().call(local_func, &[v]);

                                        let results = self.builder.inst_results(call);
                                        assert_eq!(results.len(), 1);
                                        TypedValue::F64(results[0].clone())
                                    },
                                    _ => unimplemented!()
                                }
                            },
                            ScalarFuncSig::Pow => {
                                assert_eq!(children.len(), 2);
                                let func_id = self.get_func_id("f64_pow_f64").unwrap();
                                let local_func = self.module.declare_func_in_func(func_id, &mut self.builder.func);
                                let x = children[0].value().clone();
                                let y =  children[1].value().clone();

                                let call = self.builder.ins().call(local_func, &[x, y]);
                                let results = self.builder.inst_results(call);
                                assert_eq!(results.len(), 1);
                                TypedValue::F64(results[0].clone())
                            },
                            _ => unimplemented!()
                        }
                    })
            }
            _ => unimplemented!()
        }
    }

    fn get_func_id(&self, name: &str) -> Option<FuncId> {
        self.module.get_name(name).and_then(|id|match id {
            FuncOrDataId::Func(id) => Some(id),
            FuncOrDataId::Data(_) => None
        })
    }
    fn get_data_id(&self, name: &str) -> Option<DataId> {
        self.module.get_name(name).and_then(|id|match id {
            FuncOrDataId::Func(_) => None,
            FuncOrDataId::Data(id) => Some(id)
        })
    }
}

#[cfg(test)]
mod test{
    use std::{f64, i64, u64};
    use super::*;
    use crate::coprocessor::codec::Datum;
    use crate::coprocessor::dag::expr::tests::{
        check_overflow, eval_func, eval_func_with, str2dec, scalar_func_expr, datum_expr
    };

    #[test]
    fn test_abs() {
        let tests = vec![
            (ScalarFuncSig::AbsInt, Datum::I64(-3), Datum::I64(3)),
            (
                ScalarFuncSig::AbsInt,
                Datum::I64(i64::MAX),
                Datum::I64(i64::MAX),
            ),
//            (ScalarFuncSig::AbsUInt, Datum::U64(u64::MAX), Datum::U64(u64::MAX), ),
            (ScalarFuncSig::AbsReal, Datum::F64(3.5), Datum::F64(3.5)),
            (ScalarFuncSig::AbsReal, Datum::F64(-3.5), Datum::F64(3.5)),
//            (ScalarFuncSig::AbsDecimal, str2dec("1.1"), str2dec("1.1")),
//            (ScalarFuncSig::AbsDecimal, str2dec("-1.1"), str2dec("1.1")),
        ];
        for (sig, arg, exp) in tests {
            let ctx = EvalContext::default();
            let expr = scalar_func_expr(sig, &[datum_expr(arg)]);
            let mut jit = Jit::new();
            let func: fn(&[Datum]) -> Box<Datum>  = jit.build(&ctx, expr).unwrap();

            let got = *func(&[]);
            assert_eq!(got, exp);
        }
    }


    #[test]
    fn test_pow() {
        let tests = vec![
            (Datum::F64(1.0), Datum::F64(3.0), Datum::F64(1.0)),
            (Datum::F64(3.0), Datum::F64(0.0), Datum::F64(1.0)),
            (Datum::F64(2.0), Datum::F64(4.0), Datum::F64(16.0)),
        ];
        for (arg0, arg1, exp) in tests {
            let ctx = EvalContext::default();
            let mut jit = Jit::new();
            let expr = scalar_func_expr(ScalarFuncSig::Pow, &[datum_expr(arg0), datum_expr(arg1)]);
            let func: fn(&[Datum]) -> Box<Datum> = jit.build(&ctx, expr).unwrap();
            let value = *func(&[]);
            assert_eq!(value, exp);
        }
    }

    #[test]
    fn test_pow1() {
        let tests = vec![
            (Datum::F64(1.0), Datum::F64(3.0), Datum::F64(1.0)),
            (Datum::F64(3.0), Datum::F64(0.0), Datum::F64(1.0)),
            (Datum::F64(2.0), Datum::F64(4.0), Datum::F64(16.0)),
        ];
        for (arg0, arg1, exp) in tests {
            let ctx = EvalContext::default();
            let mut jit = Jit::new();
            let expr = scalar_func_expr(ScalarFuncSig::Pow, &[datum_expr(arg0), datum_expr(arg1)]);
            let func: fn(&[Datum]) -> f64 = jit.build1(&ctx, expr).unwrap();
            let value = func(&[]);
            assert_eq!(Datum::F64(value), exp);
        }
    }

}