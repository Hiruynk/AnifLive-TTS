use mlx_rs::{Array, Device, DeviceType, Dtype, Stream, nn::{gelu, gelu_approximate, mish, silu, sigmoid}, ops::{erf,tanh}};
use std::{error::Error,fs,path::PathBuf};
fn run()->Result<(),Box<dyn Error>>{
 if !matches!(Stream::thread_local_or_default().device()?.get_type()?,DeviceType::Gpu){return Err("Metal required".into())}
 let args:Vec<_>=std::env::args().collect();if args.len()!=3{return Err("usage: fp16_activation_ops INPUT.npy FRESH_OUT".into())}
 let out=PathBuf::from(&args[2]);fs::create_dir(&out)?;
 let x=mlx_rs::with_device(Device::cpu(),||Array::load_numpy(&args[1]))?;
 if x.dtype()!=Dtype::Float16||x.shape()!=[1,619,2048]{return Err("unexpected fixture".into())}
 let f=x.as_dtype(Dtype::Float32)?;
 let exact=erf(f.multiply(Array::from_f32(0.70703125))?)?.add(Array::from_f32(1.))?.multiply(Array::from_f32(0.5))?.multiply(&f)?.as_dtype(Dtype::Float16)?;
 let cube=f.multiply(&f)?.multiply(&f)?;
 let approx=tanh(cube.multiply(Array::from_f32(0.044708251953125))?.add(&f)?.multiply(Array::from_f32(0.7978515625))?)?.add(Array::from_f32(1.))?.multiply(Array::from_f32(0.5))?.multiply(&f)?.as_dtype(Dtype::Float16)?;
 let xhalf3=x.multiply(&x)?.multiply(&x)?;
 let ah=tanh(xhalf3.multiply(Array::from_f32(0.044708251953125).as_dtype(Dtype::Float16)?)?.add(&x)?.multiply(Array::from_f32(0.7978515625).as_dtype(Dtype::Float16)?)?)?.add(Array::from_f32(1.).as_dtype(Dtype::Float16)?)?.multiply(Array::from_f32(0.5).as_dtype(Dtype::Float16)?)?.multiply(&x)?;
 let outputs=[("gelu-exact-direct",gelu(&x)?),("gelu-exact-f32",gelu(&f)?.as_dtype(Dtype::Float16)?),("gelu-exact-mps-coeff",exact),("gelu-tanh-direct",gelu_approximate(&x)?),("gelu-tanh-f32",gelu_approximate(&f)?.as_dtype(Dtype::Float16)?),("gelu-tanh-mps-coeff",approx),("gelu-tanh-mps-half",ah),("silu-direct",silu(&x)?),("silu-f32",silu(&f)?.as_dtype(Dtype::Float16)?),("mish-direct",mish(&x)?),("mish-f32",mish(&f)?.as_dtype(Dtype::Float16)?),("sigmoid-direct",sigmoid(&x)?),("sigmoid-f32",sigmoid(&f)?.as_dtype(Dtype::Float16)?)];
 mlx_rs::transforms::eval(outputs.iter().map(|(_,a)|a))?;
 for(name,a)in outputs{a.save_numpy(out.join(format!("{name}.npy")))?;}
 Ok(())
}
fn main()->Result<(),Box<dyn Error>>{mlx_rs::with_device(Device::gpu(),run)}
