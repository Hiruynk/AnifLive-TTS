use mlx_rs::{Array,Device,Dtype,nn::{mish,softplus},ops::{conv1d,tanh}};
use std::{error::Error,fs,path::PathBuf};
fn run()->Result<(),Box<dyn Error>>{
 let a:Vec<_>=std::env::args().collect();if a.len()!=3{return Err("usage: fp16_style_ops SOURCE FRESH_OUT".into())}
 let src=PathBuf::from(&a[1]);let out=PathBuf::from(&a[2]);fs::create_dir(&out)?;
 let load=|n:&str|mlx_rs::with_device(Device::cpu(),||Array::load_numpy(src.join(n)));
 let x=load("spectral.1--input-00.npy")?;let xf=x.as_dtype(Dtype::Float32)?;
 let m0=mish(&x)?;let mf=mish(&xf)?.as_dtype(Dtype::Float16)?;
 let soft=softplus(&xf)?.as_dtype(Dtype::Float16)?;
 let mc=x.multiply(tanh(&soft)?)?;
 let input=load("temporal.0.conv1.conv--input-00.npy")?.transpose_axes(&[0,2,1])?;
 let w=load("weights--temporal.0.conv1.conv.weight.npy")?.transpose_axes(&[0,2,1])?;
 let b=load("weights--temporal.0.conv1.conv.bias.npy")?;
 let ch=conv1d(&input,&w,1,2,1,1)?.add(&b)?.transpose_axes(&[0,2,1])?;
 let cf=conv1d(input.as_dtype(Dtype::Float32)?,w.as_dtype(Dtype::Float32)?,1,2,1,1)?.add(b.as_dtype(Dtype::Float32)?)?.as_dtype(Dtype::Float16)?.transpose_axes(&[0,2,1])?;
 let vals=[("mish-half",m0),("mish-f32",mf),("mish-custom-f32-softplus-half",mc),("conv-half-bias-half",ch),("conv-f32-opmath-half",cf)];
 mlx_rs::transforms::eval(vals.iter().map(|(_,v)|v))?;for(n,v)in vals{v.save_numpy(out.join(format!("{n}.npy")))?;}Ok(())
}
fn main()->Result<(),Box<dyn Error>>{mlx_rs::with_device(Device::gpu(),run)}
