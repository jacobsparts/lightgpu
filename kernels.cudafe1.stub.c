#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "kernels.fatbin.c"
extern void __device_stub__Z7lg_noopv(void);
extern void __device_stub__Z11lg_rms_normPKfS0_Pfiif(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float);
extern void __device_stub__Z13lg_layer_normPKfS0_S0_Pfiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float);
extern void __device_stub__Z19lg_layer_norm_2passPKfS0_S0_Pfiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float);
extern void __device_stub__Z12lg_rope_neoxPfPKiiiif(float *__restrict__, const int *__restrict__, int, int, int, float);
extern void __device_stub__Z10lg_rope_2dPfPKfS1_iii(float *__restrict__, const float *__restrict__, const float *__restrict__, int, int, int);
extern void __device_stub__Z11lg_silu_mulPfPKfi(float *__restrict__, const float *__restrict__, int);
extern void __device_stub__Z12lg_gelu_tanhPKfPfi(const float *__restrict__, float *__restrict__, int);
extern void __device_stub__Z11lg_gelu_erfPKfPfi(const float *__restrict__, float *__restrict__, int);
extern void __device_stub__Z7lg_reluPKfPfi(const float *__restrict__, float *__restrict__, int);
extern void __device_stub__Z10lg_sigmoidPKfPfi(const float *__restrict__, float *__restrict__, int);
extern void __device_stub__Z6lg_addPKfS0_Pfi(const float *__restrict__, const float *__restrict__, float *__restrict__, int);
extern void __device_stub__Z6lg_mulPKfS0_Pfi(const float *__restrict__, const float *__restrict__, float *__restrict__, int);
extern void __device_stub__Z14lg_add_inplacePfPKfi(float *__restrict__, const float *__restrict__, int);
extern void __device_stub__Z8lg_scalePKfPffi(const float *__restrict__, float *__restrict__, float, int);
extern void __device_stub__Z13lg_row_affinePfPKfS1_ii(float *__restrict__, const float *__restrict__, const float *__restrict__, int, int);
extern void __device_stub__Z21lg_channel_layer_normPKfS0_S0_Pfiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float);
extern void __device_stub__Z17lg_channel_affinePKfPfS0_S0_ii(const float *__restrict__, float *__restrict__, const float *__restrict__, const float *__restrict__, int, int);
extern void __device_stub__Z16lg_channel_scalePKfS0_Pfii(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int);
extern void __device_stub__Z7lg_copyPKfPfl(const float *__restrict__, float *__restrict__, long);
extern void __device_stub__Z15lg_channel_meanPKfPfii(const float *__restrict__, float *__restrict__, int, int);
extern void __device_stub__Z9lg_argmaxPKfPiPfii(const float *__restrict__, int *__restrict__, float *__restrict__, int, int);
extern void __device_stub__Z10lg_conv1x1PKfS0_S0_Pfiiii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int);
extern void __device_stub__Z14lg_conv3x3s1p1PKfS0_S0_Pfiiii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int);
extern void __device_stub__Z12lg_conv4x4s4PKfS0_S0_Pfiiii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int);
extern void __device_stub__Z11lg_conv_kxkPKfS0_S0_Pfiiiii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int);
extern void __device_stub__Z13lg_linear_1x1PKfS0_S0_Pfii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int);
extern void __device_stub__Z19lg_conv3x3_winogradPKfS0_S0_Pfiiiiiiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, int, int, float);
extern void __device_stub__Z15lg_extract_rowsPfPKfiii(float *__restrict__, const float *__restrict__, int, int, int);
extern void __device_stub__Z12lg_merge_2x2PfPKfiii(float *__restrict__, const float *__restrict__, int, int, int);
extern void __device_stub__Z23lg_get_row_q8_0_alignedPKhiPfi(const uint8_t *__restrict__, int, float *__restrict__, int);
extern void __device_stub__Z11lg_copy_rowPK6float4PS_i(const struct float4 *__restrict__, struct float4 *__restrict__, int);
extern void __device_stub__Z10lg_set_i32Pii(int *__restrict__, int);
extern void __device_stub__Z11lg_attn_gqaPKfS0_S0_S0_Pfiiiiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float);
extern void __device_stub__Z13lg_attn_flashPKfS0_S0_S0_Pfiiiifi(const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, float, int);
extern void __device_stub__Z22lg_attn_prefill_scoresPKfS0_S0_Pfiiiiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float);
extern void __device_stub__Z21lg_attn_prefill_fusedPKfS0_S0_S0_Pfiiiiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float);
extern void __device_stub__Z23lg_attn_prefill_softmaxPfii(float *__restrict__, int, int);
extern void __device_stub__Z19lg_attn_prefill_outPKfS0_Pfiiiii(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int);
extern void __device_stub__Z9lg_linearPKfS0_S0_Pfiii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int);
extern void __device_stub__Z11lg_f32_gemmPKfS0_Pfiii(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int);
extern void __device_stub__Z17lg_f32_gemm_tiledPKfS0_Pfiii(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int);
extern void __device_stub__Z16lg_quantize_q8_0PKfPaPfii(const float *__restrict__, int8_t *__restrict__, float *__restrict__, int, int);
extern void __device_stub__Z17lg_q8_0_gemm_dp4aPKhPKaPKfPfiii(const uint8_t *__restrict__, const int8_t *__restrict__, const float *__restrict__, float *__restrict__, int, int, int);
extern void __device_stub__Z20lg_q8_0_gemm_alignedPKhPKfPfiii(const uint8_t *__restrict__, const float *__restrict__, float *__restrict__, int, int, int);
extern void __device_stub__Z12lg_q8_0_gemvPKhPKaPKfPfiiiS4_(const uint8_t *__restrict__, const int8_t *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, const float *__restrict__);
extern void __device_stub__Z8lg_lreluPKfPffl(const float *__restrict__, float *__restrict__, float, long);
extern void __device_stub__Z13lg_add_scaledPKfS0_Pflf(const float *__restrict__, const float *__restrict__, float *__restrict__, long, float);
extern void __device_stub__Z21lg_upsample2x_nearestPKfPfiii(const float *__restrict__, float *__restrict__, int, int, int);
extern void __device_stub__Z19lg_pixel_unshuffle2PKfPfiii(const float *__restrict__, float *__restrict__, int, int, int);
extern void __device_stub__Z16lg_pixel_shufflePKfPfiiii(const float *__restrict__, float *__restrict__, int, int, int, int);
extern void __device_stub__Z11lg_fft2_r2cPKfPfiiif(const float *__restrict__, float *__restrict__, int, int, int, float);
extern void __device_stub__Z11lg_fft2_c2rPKfPfiiif(const float *__restrict__, float *__restrict__, int, int, int, float);
extern void __device_stub__Z12lg_linear_rbPKfS0_S0_Pfiii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int);
extern void __device_stub__Z13lg_conv1x1_rbPKfS0_S0_Pfiiii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int);
extern void __device_stub__Z18lg_layer_norm_warpPKfS0_S0_Pfiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float);
extern void __device_stub__Z15lg_conv3x3_tilePKfS0_S0_Pfiiiiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float);
extern void __device_stub__Z15lg_conv1x1_tilePKfS0_S0_Pfiiiiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float);
extern void __device_stub__Z12lg_conv2x2s2PKfS0_S0_Pfiiii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int);
extern void __device_stub__Z12lg_conv_t2x2PKfS0_S0_Pfiiii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int);
extern void __device_stub__Z16lg_window_gatherPKfPfiiiiiiiii(const float *__restrict__, float *__restrict__, int, int, int, int, int, int, int, int, int);
extern void __device_stub__Z17lg_window_scatterPKfPfiiiiiiiii(const float *__restrict__, float *__restrict__, int, int, int, int, int, int, int, int, int);
extern void __device_stub__Z13lg_conv3x3_q2PKfS0_S0_Pfiiiiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float);
extern void __device_stub__Z16lg_conv3x3_q2ng2PKfS0_S0_Pfiiiiif(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float);
extern void __device_stub__Z16lg_conv3x3_catq0PKfS0_S0_PfiiiiifS0_S0_S0_S0_S0_iiiiii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, int, int, int, int, int, int);
extern void __device_stub__Z19lg_conv3x3_catq0ng2PKfS0_S0_PfiiiiifS0_S0_S0_S0_S0_iiiiii(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, int, int, int, int, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z7lg_noopv(void){__cudaLaunchPrologue(1);__cudaLaunch(((char *)((void ( *)(void))lg_noop)));}
# 40 "cuda/kernels.cu"
void lg_noop(void)
# 40 "cuda/kernels.cu"
{__device_stub__Z7lg_noopv(); }
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z11lg_rms_normPKfS0_Pfiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  float *__restrict__ __par2,  int __par3,  int __par4,  float __par5) {  const float *__T14;
 const float *__T15;
 float *__T16;
__cudaLaunchPrologue(6); __T14 = __par0; __cudaSetupArgSimple(__T14, 0UL); __T15 = __par1; __cudaSetupArgSimple(__T15, 8UL); __T16 = __par2; __cudaSetupArgSimple(__T16, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaSetupArgSimple(__par5, 32UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float))lg_rms_norm))); }
# 47 "cuda/kernels.cu"
void lg_rms_norm( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,float *__restrict__ __cuda_2,int __cuda_3,int __cuda_4,float __cuda_5)
# 50 "cuda/kernels.cu"
{__device_stub__Z11lg_rms_normPKfS0_Pfiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 66 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z13lg_layer_normPKfS0_S0_Pfiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  float __par6) {  const float *__T17;
 const float *__T18;
 const float *__T19;
 float *__T20;
__cudaLaunchPrologue(7); __T17 = __par0; __cudaSetupArgSimple(__T17, 0UL); __T18 = __par1; __cudaSetupArgSimple(__T18, 8UL); __T19 = __par2; __cudaSetupArgSimple(__T19, 16UL); __T20 = __par3; __cudaSetupArgSimple(__T20, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float))lg_layer_norm))); }
# 79 "cuda/kernels.cu"
void lg_layer_norm( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,float __cuda_6)
# 82 "cuda/kernels.cu"
{__device_stub__Z13lg_layer_normPKfS0_S0_Pfiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 109 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z19lg_layer_norm_2passPKfS0_S0_Pfiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  float __par6) {  const float *__T21;
 const float *__T22;
 const float *__T23;
 float *__T24;
__cudaLaunchPrologue(7); __T21 = __par0; __cudaSetupArgSimple(__T21, 0UL); __T22 = __par1; __cudaSetupArgSimple(__T22, 8UL); __T23 = __par2; __cudaSetupArgSimple(__T23, 16UL); __T24 = __par3; __cudaSetupArgSimple(__T24, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float))lg_layer_norm_2pass))); }
# 114 "cuda/kernels.cu"
void lg_layer_norm_2pass( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,float __cuda_6)
# 117 "cuda/kernels.cu"
{__device_stub__Z19lg_layer_norm_2passPKfS0_S0_Pfiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 146 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z12lg_rope_neoxPfPKiiiif( float *__restrict__ __par0,  const int *__restrict__ __par1,  int __par2,  int __par3,  int __par4,  float __par5) {  float *__T25;
 const int *__T26;
__cudaLaunchPrologue(6); __T25 = __par0; __cudaSetupArgSimple(__T25, 0UL); __T26 = __par1; __cudaSetupArgSimple(__T26, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 28UL); __cudaLaunch(((char *)((void ( *)(float *__restrict__, const int *__restrict__, int, int, int, float))lg_rope_neox))); }
# 154 "cuda/kernels.cu"
void lg_rope_neox( float *__restrict__ __cuda_0,const int *__restrict__ __cuda_1,int __cuda_2,int __cuda_3,int __cuda_4,float __cuda_5)
# 157 "cuda/kernels.cu"
{__device_stub__Z12lg_rope_neoxPfPKiiiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 172 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z10lg_rope_2dPfPKfS1_iii( float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  int __par3,  int __par4,  int __par5) {  float *__T27;
 const float *__T28;
 const float *__T29;
__cudaLaunchPrologue(6); __T27 = __par0; __cudaSetupArgSimple(__T27, 0UL); __T28 = __par1; __cudaSetupArgSimple(__T28, 8UL); __T29 = __par2; __cudaSetupArgSimple(__T29, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaSetupArgSimple(__par5, 32UL); __cudaLaunch(((char *)((void ( *)(float *__restrict__, const float *__restrict__, const float *__restrict__, int, int, int))lg_rope_2d))); }
# 177 "cuda/kernels.cu"
void lg_rope_2d( float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,int __cuda_3,int __cuda_4,int __cuda_5)
# 180 "cuda/kernels.cu"
{__device_stub__Z10lg_rope_2dPfPKfS1_iii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 193 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z11lg_silu_mulPfPKfi( float *__restrict__ __par0,  const float *__restrict__ __par1,  int __par2) {  float *__T30;
 const float *__T31;
__cudaLaunchPrologue(3); __T30 = __par0; __cudaSetupArgSimple(__T30, 0UL); __T31 = __par1; __cudaSetupArgSimple(__T31, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *__restrict__, const float *__restrict__, int))lg_silu_mul))); }
# 207 "cuda/kernels.cu"
void lg_silu_mul( float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,int __cuda_2)
# 209 "cuda/kernels.cu"
{__device_stub__Z11lg_silu_mulPfPKfi( __cuda_0,__cuda_1,__cuda_2);


}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z12lg_gelu_tanhPKfPfi( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2) {  const float *__T32;
 float *__T33;
__cudaLaunchPrologue(3); __T32 = __par0; __cudaSetupArgSimple(__T32, 0UL); __T33 = __par1; __cudaSetupArgSimple(__T33, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int))lg_gelu_tanh))); }
# 215 "cuda/kernels.cu"
void lg_gelu_tanh( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2)
# 217 "cuda/kernels.cu"
{__device_stub__Z12lg_gelu_tanhPKfPfi( __cuda_0,__cuda_1,__cuda_2);
# 223 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z11lg_gelu_erfPKfPfi( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2) {  const float *__T34;
 float *__T35;
__cudaLaunchPrologue(3); __T34 = __par0; __cudaSetupArgSimple(__T34, 0UL); __T35 = __par1; __cudaSetupArgSimple(__T35, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int))lg_gelu_erf))); }
# 244 "cuda/kernels.cu"
void lg_gelu_erf( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2)
# 246 "cuda/kernels.cu"
{__device_stub__Z11lg_gelu_erfPKfPfi( __cuda_0,__cuda_1,__cuda_2);




}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z7lg_reluPKfPfi( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2) {  const float *__T36;
 float *__T37;
__cudaLaunchPrologue(3); __T36 = __par0; __cudaSetupArgSimple(__T36, 0UL); __T37 = __par1; __cudaSetupArgSimple(__T37, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int))lg_relu))); }
# 255 "cuda/kernels.cu"
void lg_relu( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2)
# 257 "cuda/kernels.cu"
{__device_stub__Z7lg_reluPKfPfi( __cuda_0,__cuda_1,__cuda_2);




}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z10lg_sigmoidPKfPfi( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2) {  const float *__T38;
 float *__T39;
__cudaLaunchPrologue(3); __T38 = __par0; __cudaSetupArgSimple(__T38, 0UL); __T39 = __par1; __cudaSetupArgSimple(__T39, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int))lg_sigmoid))); }
# 272 "cuda/kernels.cu"
void lg_sigmoid( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2)
# 274 "cuda/kernels.cu"
{__device_stub__Z10lg_sigmoidPKfPfi( __cuda_0,__cuda_1,__cuda_2);



}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z6lg_addPKfS0_Pfi( const float *__restrict__ __par0,  const float *__restrict__ __par1,  float *__restrict__ __par2,  int __par3) {  const float *__T40;
 const float *__T41;
 float *__T42;
__cudaLaunchPrologue(4); __T40 = __par0; __cudaSetupArgSimple(__T40, 0UL); __T41 = __par1; __cudaSetupArgSimple(__T41, 8UL); __T42 = __par2; __cudaSetupArgSimple(__T42, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int))lg_add))); }
# 281 "cuda/kernels.cu"
void lg_add( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,float *__restrict__ __cuda_2,int __cuda_3)
# 283 "cuda/kernels.cu"
{__device_stub__Z6lg_addPKfS0_Pfi( __cuda_0,__cuda_1,__cuda_2,__cuda_3);


}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z6lg_mulPKfS0_Pfi( const float *__restrict__ __par0,  const float *__restrict__ __par1,  float *__restrict__ __par2,  int __par3) {  const float *__T43;
 const float *__T44;
 float *__T45;
__cudaLaunchPrologue(4); __T43 = __par0; __cudaSetupArgSimple(__T43, 0UL); __T44 = __par1; __cudaSetupArgSimple(__T44, 8UL); __T45 = __par2; __cudaSetupArgSimple(__T45, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int))lg_mul))); }
# 302 "cuda/kernels.cu"
void lg_mul( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,float *__restrict__ __cuda_2,int __cuda_3)
# 304 "cuda/kernels.cu"
{__device_stub__Z6lg_mulPKfS0_Pfi( __cuda_0,__cuda_1,__cuda_2,__cuda_3);


}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z14lg_add_inplacePfPKfi( float *__restrict__ __par0,  const float *__restrict__ __par1,  int __par2) {  float *__T46;
 const float *__T47;
__cudaLaunchPrologue(3); __T46 = __par0; __cudaSetupArgSimple(__T46, 0UL); __T47 = __par1; __cudaSetupArgSimple(__T47, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *__restrict__, const float *__restrict__, int))lg_add_inplace))); }
# 312 "cuda/kernels.cu"
void lg_add_inplace( float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,int __cuda_2)
# 313 "cuda/kernels.cu"
{__device_stub__Z14lg_add_inplacePfPKfi( __cuda_0,__cuda_1,__cuda_2);


}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z8lg_scalePKfPffi( const float *__restrict__ __par0,  float *__restrict__ __par1,  float __par2,  int __par3) {  const float *__T48;
 float *__T49;
__cudaLaunchPrologue(4); __T48 = __par0; __cudaSetupArgSimple(__T48, 0UL); __T49 = __par1; __cudaSetupArgSimple(__T49, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, float, int))lg_scale))); }
# 319 "cuda/kernels.cu"
void lg_scale( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,float __cuda_2,int __cuda_3)
# 321 "cuda/kernels.cu"
{__device_stub__Z8lg_scalePKfPffi( __cuda_0,__cuda_1,__cuda_2,__cuda_3);


}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z13lg_row_affinePfPKfS1_ii( float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  int __par3,  int __par4) {  float *__T50;
 const float *__T51;
 const float *__T52;
__cudaLaunchPrologue(5); __T50 = __par0; __cudaSetupArgSimple(__T50, 0UL); __T51 = __par1; __cudaSetupArgSimple(__T51, 8UL); __T52 = __par2; __cudaSetupArgSimple(__T52, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaLaunch(((char *)((void ( *)(float *__restrict__, const float *__restrict__, const float *__restrict__, int, int))lg_row_affine))); }
# 335 "cuda/kernels.cu"
void lg_row_affine( float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,int __cuda_3,int __cuda_4)
# 338 "cuda/kernels.cu"
{__device_stub__Z13lg_row_affinePfPKfS1_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 349 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z21lg_channel_layer_normPKfS0_S0_Pfiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  float __par6) {  const float *__T53;
 const float *__T54;
 const float *__T55;
 float *__T56;
__cudaLaunchPrologue(7); __T53 = __par0; __cudaSetupArgSimple(__T53, 0UL); __T54 = __par1; __cudaSetupArgSimple(__T54, 8UL); __T55 = __par2; __cudaSetupArgSimple(__T55, 16UL); __T56 = __par3; __cudaSetupArgSimple(__T56, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float))lg_channel_layer_norm))); }
# 385 "cuda/kernels.cu"
void lg_channel_layer_norm( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,float __cuda_6)
# 388 "cuda/kernels.cu"
{__device_stub__Z21lg_channel_layer_normPKfS0_S0_Pfiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 409 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z17lg_channel_affinePKfPfS0_S0_ii( const float *__restrict__ __par0,  float *__restrict__ __par1,  const float *__restrict__ __par2,  const float *__restrict__ __par3,  int __par4,  int __par5) {  const float *__T57;
 float *__T58;
 const float *__T59;
 const float *__T60;
__cudaLaunchPrologue(6); __T57 = __par0; __cudaSetupArgSimple(__T57, 0UL); __T58 = __par1; __cudaSetupArgSimple(__T58, 8UL); __T59 = __par2; __cudaSetupArgSimple(__T59, 16UL); __T60 = __par3; __cudaSetupArgSimple(__T60, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, const float *__restrict__, const float *__restrict__, int, int))lg_channel_affine))); }
# 423 "cuda/kernels.cu"
void lg_channel_affine( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,const float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5)
# 427 "cuda/kernels.cu"
{__device_stub__Z17lg_channel_affinePKfPfS0_S0_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 435 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z16lg_channel_scalePKfS0_Pfii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  float *__restrict__ __par2,  int __par3,  int __par4) {  const float *__T61;
 const float *__T62;
 float *__T63;
__cudaLaunchPrologue(5); __T61 = __par0; __cudaSetupArgSimple(__T61, 0UL); __T62 = __par1; __cudaSetupArgSimple(__T62, 8UL); __T63 = __par2; __cudaSetupArgSimple(__T63, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int))lg_channel_scale))); }
# 447 "cuda/kernels.cu"
void lg_channel_scale( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,float *__restrict__ __cuda_2,int __cuda_3,int __cuda_4)
# 450 "cuda/kernels.cu"
{__device_stub__Z16lg_channel_scalePKfS0_Pfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 456 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z7lg_copyPKfPfl( const float *__restrict__ __par0,  float *__restrict__ __par1,  long __par2) {  const float *__T64;
 float *__T65;
__cudaLaunchPrologue(3); __T64 = __par0; __cudaSetupArgSimple(__T64, 0UL); __T65 = __par1; __cudaSetupArgSimple(__T65, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, long))lg_copy))); }
# 460 "cuda/kernels.cu"
void lg_copy( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,long __cuda_2)
# 462 "cuda/kernels.cu"
{__device_stub__Z7lg_copyPKfPfl( __cuda_0,__cuda_1,__cuda_2);


}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z15lg_channel_meanPKfPfii( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2,  int __par3) {  const float *__T66;
 float *__T67;
__cudaLaunchPrologue(4); __T66 = __par0; __cudaSetupArgSimple(__T66, 0UL); __T67 = __par1; __cudaSetupArgSimple(__T67, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int, int))lg_channel_mean))); }
# 498 "cuda/kernels.cu"
void lg_channel_mean( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2,int __cuda_3)
# 500 "cuda/kernels.cu"
{__device_stub__Z15lg_channel_meanPKfPfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 516 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z9lg_argmaxPKfPiPfii( const float *__restrict__ __par0,  int *__restrict__ __par1,  float *__restrict__ __par2,  int __par3,  int __par4) {  const float *__T68;
 int *__T69;
 float *__T70;
__cudaLaunchPrologue(5); __T68 = __par0; __cudaSetupArgSimple(__T68, 0UL); __T69 = __par1; __cudaSetupArgSimple(__T69, 8UL); __T70 = __par2; __cudaSetupArgSimple(__T70, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, int *__restrict__, float *__restrict__, int, int))lg_argmax))); }
# 520 "cuda/kernels.cu"
void lg_argmax( const float *__restrict__ __cuda_0,int *__restrict__ __cuda_1,float *__restrict__ __cuda_2,int __cuda_3,int __cuda_4)
# 523 "cuda/kernels.cu"
{__device_stub__Z9lg_argmaxPKfPiPfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 543 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z10lg_conv1x1PKfS0_S0_Pfiiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7) {  const float *__T71;
 const float *__T72;
 const float *__T73;
 float *__T74;
__cudaLaunchPrologue(8); __T71 = __par0; __cudaSetupArgSimple(__T71, 0UL); __T72 = __par1; __cudaSetupArgSimple(__T72, 8UL); __T73 = __par2; __cudaSetupArgSimple(__T73, 16UL); __T74 = __par3; __cudaSetupArgSimple(__T74, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv1x1))); }
# 555 "cuda/kernels.cu"
void lg_conv1x1( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7)
# 559 "cuda/kernels.cu"
{__device_stub__Z10lg_conv1x1PKfS0_S0_Pfiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 573 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z14lg_conv3x3s1p1PKfS0_S0_Pfiiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7) {  const float *__T75;
 const float *__T76;
 const float *__T77;
 float *__T78;
__cudaLaunchPrologue(8); __T75 = __par0; __cudaSetupArgSimple(__T75, 0UL); __T76 = __par1; __cudaSetupArgSimple(__T76, 8UL); __T77 = __par2; __cudaSetupArgSimple(__T77, 16UL); __T78 = __par3; __cudaSetupArgSimple(__T78, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv3x3s1p1))); }
# 576 "cuda/kernels.cu"
void lg_conv3x3s1p1( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7)
# 580 "cuda/kernels.cu"
{__device_stub__Z14lg_conv3x3s1p1PKfS0_S0_Pfiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 606 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z12lg_conv4x4s4PKfS0_S0_Pfiiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7) {  const float *__T79;
 const float *__T80;
 const float *__T81;
 float *__T82;
__cudaLaunchPrologue(8); __T79 = __par0; __cudaSetupArgSimple(__T79, 0UL); __T80 = __par1; __cudaSetupArgSimple(__T80, 8UL); __T81 = __par2; __cudaSetupArgSimple(__T81, 16UL); __T82 = __par3; __cudaSetupArgSimple(__T82, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv4x4s4))); }
# 609 "cuda/kernels.cu"
void lg_conv4x4s4( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7)
# 613 "cuda/kernels.cu"
{__device_stub__Z12lg_conv4x4s4PKfS0_S0_Pfiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 635 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z11lg_conv_kxkPKfS0_S0_Pfiiiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7,  int __par8) {  const float *__T83;
 const float *__T84;
 const float *__T85;
 float *__T86;
__cudaLaunchPrologue(9); __T83 = __par0; __cudaSetupArgSimple(__T83, 0UL); __T84 = __par1; __cudaSetupArgSimple(__T84, 8UL); __T85 = __par2; __cudaSetupArgSimple(__T85, 16UL); __T86 = __par3; __cudaSetupArgSimple(__T86, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaSetupArgSimple(__par8, 48UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int))lg_conv_kxk))); }
# 638 "cuda/kernels.cu"
void lg_conv_kxk( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8)
# 642 "cuda/kernels.cu"
{__device_stub__Z11lg_conv_kxkPKfS0_S0_Pfiiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8);
# 670 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z13lg_linear_1x1PKfS0_S0_Pfii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5) {  const float *__T87;
 const float *__T88;
 const float *__T89;
 float *__T90;
__cudaLaunchPrologue(6); __T87 = __par0; __cudaSetupArgSimple(__T87, 0UL); __T88 = __par1; __cudaSetupArgSimple(__T88, 8UL); __T89 = __par2; __cudaSetupArgSimple(__T89, 16UL); __T90 = __par3; __cudaSetupArgSimple(__T90, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int))lg_linear_1x1))); }
# 673 "cuda/kernels.cu"
void lg_linear_1x1( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5)
# 677 "cuda/kernels.cu"
{__device_stub__Z13lg_linear_1x1PKfS0_S0_Pfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 684 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z19lg_conv3x3_winogradPKfS0_S0_Pfiiiiiiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7,  int __par8,  int __par9,  int __par10,  float __par11) {  const float *__T97;
 const float *__T98;
 const float *__T99;
 float *__T100;
__cudaLaunchPrologue(12); __T97 = __par0; __cudaSetupArgSimple(__T97, 0UL); __T98 = __par1; __cudaSetupArgSimple(__T98, 8UL); __T99 = __par2; __cudaSetupArgSimple(__T99, 16UL); __T100 = __par3; __cudaSetupArgSimple(__T100, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaSetupArgSimple(__par8, 48UL); __cudaSetupArgSimple(__par9, 52UL); __cudaSetupArgSimple(__par10, 56UL); __cudaSetupArgSimple(__par11, 60UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, int, int, float))lg_conv3x3_winograd))); }
# 940 "cuda/kernels.cu"
void lg_conv3x3_winograd( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,int __cuda_9,int __cuda_10,float __cuda_11)
# 945 "cuda/kernels.cu"
{__device_stub__Z19lg_conv3x3_winogradPKfS0_S0_Pfiiiiiiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10,__cuda_11);

}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z15lg_extract_rowsPfPKfiii( float *__restrict__ __par0,  const float *__restrict__ __par1,  int __par2,  int __par3,  int __par4) {  float *__T101;
 const float *__T102;
__cudaLaunchPrologue(5); __T101 = __par0; __cudaSetupArgSimple(__T101, 0UL); __T102 = __par1; __cudaSetupArgSimple(__T102, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaLaunch(((char *)((void ( *)(float *__restrict__, const float *__restrict__, int, int, int))lg_extract_rows))); }
# 961 "cuda/kernels.cu"
void lg_extract_rows( float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,int __cuda_2,int __cuda_3,int __cuda_4)
# 964 "cuda/kernels.cu"
{__device_stub__Z15lg_extract_rowsPfPKfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 971 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z12lg_merge_2x2PfPKfiii( float *__restrict__ __par0,  const float *__restrict__ __par1,  int __par2,  int __par3,  int __par4) {  float *__T103;
 const float *__T104;
__cudaLaunchPrologue(5); __T103 = __par0; __cudaSetupArgSimple(__T103, 0UL); __T104 = __par1; __cudaSetupArgSimple(__T104, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaLaunch(((char *)((void ( *)(float *__restrict__, const float *__restrict__, int, int, int))lg_merge_2x2))); }
# 976 "cuda/kernels.cu"
void lg_merge_2x2( float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,int __cuda_2,int __cuda_3,int __cuda_4)
# 979 "cuda/kernels.cu"
{__device_stub__Z12lg_merge_2x2PfPKfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 990 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z23lg_get_row_q8_0_alignedPKhiPfi( const uint8_t *__restrict__ __par0,  int __par1,  float *__restrict__ __par2,  int __par3) {  const uint8_t *__T105;
 float *__T106;
__cudaLaunchPrologue(4); __T105 = __par0; __cudaSetupArgSimple(__T105, 0UL); __cudaSetupArgSimple(__par1, 8UL); __T106 = __par2; __cudaSetupArgSimple(__T106, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(const uint8_t *__restrict__, int, float *__restrict__, int))lg_get_row_q8_0_aligned))); }
# 995 "cuda/kernels.cu"
void lg_get_row_q8_0_aligned( const uint8_t *__restrict__ __cuda_0,int __cuda_1,float *__restrict__ __cuda_2,int __cuda_3)
# 997 "cuda/kernels.cu"
{__device_stub__Z23lg_get_row_q8_0_alignedPKhiPfi( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 1007 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z11lg_copy_rowPK6float4PS_i( const struct float4 *__restrict__ __par0,  struct float4 *__restrict__ __par1,  int __par2) {  const struct float4 *__T107;
 struct float4 *__T108;
__cudaLaunchPrologue(3); __T107 = __par0; __cudaSetupArgSimple(__T107, 0UL); __T108 = __par1; __cudaSetupArgSimple(__T108, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(const struct float4 *__restrict__, struct float4 *__restrict__, int))lg_copy_row))); }
# 1011 "cuda/kernels.cu"
void lg_copy_row( const struct float4 *__restrict__ __cuda_0,struct float4 *__restrict__ __cuda_1,int __cuda_2)
# 1012 "cuda/kernels.cu"
{__device_stub__Z11lg_copy_rowPK6float4PS_i( __cuda_0,__cuda_1,__cuda_2);



}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z10lg_set_i32Pii( int *__restrict__ __par0,  int __par1) {  int *__T109;
__cudaLaunchPrologue(2); __T109 = __par0; __cudaSetupArgSimple(__T109, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaLaunch(((char *)((void ( *)(int *__restrict__, int))lg_set_i32))); }
# 1020 "cuda/kernels.cu"
void lg_set_i32( int *__restrict__ __cuda_0,int __cuda_1)
# 1021 "cuda/kernels.cu"
{__device_stub__Z10lg_set_i32Pii( __cuda_0,__cuda_1);

}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z11lg_attn_gqaPKfS0_S0_S0_Pfiiiiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  const float *__restrict__ __par3,  float *__restrict__ __par4,  int __par5,  int __par6,  int __par7,  int __par8,  int __par9,  float __par10) {  const float *__T110;
 const float *__T111;
 const float *__T112;
 const float *__T113;
 float *__T114;
__cudaLaunchPrologue(11); __T110 = __par0; __cudaSetupArgSimple(__T110, 0UL); __T111 = __par1; __cudaSetupArgSimple(__T111, 8UL); __T112 = __par2; __cudaSetupArgSimple(__T112, 16UL); __T113 = __par3; __cudaSetupArgSimple(__T113, 24UL); __T114 = __par4; __cudaSetupArgSimple(__T114, 32UL); __cudaSetupArgSimple(__par5, 40UL); __cudaSetupArgSimple(__par6, 44UL); __cudaSetupArgSimple(__par7, 48UL); __cudaSetupArgSimple(__par8, 52UL); __cudaSetupArgSimple(__par9, 56UL); __cudaSetupArgSimple(__par10, 60UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_attn_gqa))); }
# 1042 "cuda/kernels.cu"
void lg_attn_gqa( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,const float *__restrict__ __cuda_3,float *__restrict__ __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,int __cuda_9,float __cuda_10)
# 1046 "cuda/kernels.cu"
{__device_stub__Z11lg_attn_gqaPKfS0_S0_S0_Pfiiiiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10);
# 1102 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z13lg_attn_flashPKfS0_S0_S0_Pfiiiifi( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  const float *__restrict__ __par3,  float *__restrict__ __par4,  int __par5,  int __par6,  int __par7,  int __par8,  float __par9,  int __par10) {  const float *__T115;
 const float *__T116;
 const float *__T117;
 const float *__T118;
 float *__T119;
__cudaLaunchPrologue(11); __T115 = __par0; __cudaSetupArgSimple(__T115, 0UL); __T116 = __par1; __cudaSetupArgSimple(__T116, 8UL); __T117 = __par2; __cudaSetupArgSimple(__T117, 16UL); __T118 = __par3; __cudaSetupArgSimple(__T118, 24UL); __T119 = __par4; __cudaSetupArgSimple(__T119, 32UL); __cudaSetupArgSimple(__par5, 40UL); __cudaSetupArgSimple(__par6, 44UL); __cudaSetupArgSimple(__par7, 48UL); __cudaSetupArgSimple(__par8, 52UL); __cudaSetupArgSimple(__par9, 56UL); __cudaSetupArgSimple(__par10, 60UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, float, int))lg_attn_flash))); }
# 1113 "cuda/kernels.cu"
void lg_attn_flash( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,const float *__restrict__ __cuda_3,float *__restrict__ __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,float __cuda_9,int __cuda_10)
# 1117 "cuda/kernels.cu"
{__device_stub__Z13lg_attn_flashPKfS0_S0_S0_Pfiiiifi( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10);
# 1182 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z22lg_attn_prefill_scoresPKfS0_S0_Pfiiiiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7,  int __par8,  float __par9) {  const float *__T120;
 const float *__T121;
 const float *__T122;
 float *__T123;
__cudaLaunchPrologue(10); __T120 = __par0; __cudaSetupArgSimple(__T120, 0UL); __T121 = __par1; __cudaSetupArgSimple(__T121, 8UL); __T122 = __par2; __cudaSetupArgSimple(__T122, 16UL); __T123 = __par3; __cudaSetupArgSimple(__T123, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaSetupArgSimple(__par8, 48UL); __cudaSetupArgSimple(__par9, 52UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_attn_prefill_scores))); }
# 1229 "cuda/kernels.cu"
void lg_attn_prefill_scores( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,float __cuda_9)
# 1235 "cuda/kernels.cu"
{__device_stub__Z22lg_attn_prefill_scoresPKfS0_S0_Pfiiiiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9);
# 1307 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z21lg_attn_prefill_fusedPKfS0_S0_S0_Pfiiiiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  const float *__restrict__ __par3,  float *__restrict__ __par4,  int __par5,  int __par6,  int __par7,  int __par8,  int __par9,  float __par10) {  const float *__T124;
 const float *__T125;
 const float *__T126;
 const float *__T127;
 float *__T128;
__cudaLaunchPrologue(11); __T124 = __par0; __cudaSetupArgSimple(__T124, 0UL); __T125 = __par1; __cudaSetupArgSimple(__T125, 8UL); __T126 = __par2; __cudaSetupArgSimple(__T126, 16UL); __T127 = __par3; __cudaSetupArgSimple(__T127, 24UL); __T128 = __par4; __cudaSetupArgSimple(__T128, 32UL); __cudaSetupArgSimple(__par5, 40UL); __cudaSetupArgSimple(__par6, 44UL); __cudaSetupArgSimple(__par7, 48UL); __cudaSetupArgSimple(__par8, 52UL); __cudaSetupArgSimple(__par9, 56UL); __cudaSetupArgSimple(__par10, 60UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_attn_prefill_fused))); }
# 1351 "cuda/kernels.cu"
void lg_attn_prefill_fused( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,const float *__restrict__ __cuda_3,float *__restrict__ __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,int __cuda_9,float __cuda_10)
# 1358 "cuda/kernels.cu"
{__device_stub__Z21lg_attn_prefill_fusedPKfS0_S0_S0_Pfiiiiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10);
# 1483 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z23lg_attn_prefill_softmaxPfii( float *__restrict__ __par0,  int __par1,  int __par2) {  float *__T129;
__cudaLaunchPrologue(3); __T129 = __par0; __cudaSetupArgSimple(__T129, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 12UL); __cudaLaunch(((char *)((void ( *)(float *__restrict__, int, int))lg_attn_prefill_softmax))); }
# 1490 "cuda/kernels.cu"
void lg_attn_prefill_softmax( float *__restrict__ __cuda_0,int __cuda_1,int __cuda_2)
# 1492 "cuda/kernels.cu"
{__device_stub__Z23lg_attn_prefill_softmaxPfii( __cuda_0,__cuda_1,__cuda_2);
# 1531 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z19lg_attn_prefill_outPKfS0_Pfiiiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  float *__restrict__ __par2,  int __par3,  int __par4,  int __par5,  int __par6,  int __par7) {  const float *__T130;
 const float *__T131;
 float *__T132;
__cudaLaunchPrologue(8); __T130 = __par0; __cudaSetupArgSimple(__T130, 0UL); __T131 = __par1; __cudaSetupArgSimple(__T131, 8UL); __T132 = __par2; __cudaSetupArgSimple(__T132, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaSetupArgSimple(__par5, 32UL); __cudaSetupArgSimple(__par6, 36UL); __cudaSetupArgSimple(__par7, 40UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int))lg_attn_prefill_out))); }
# 1538 "cuda/kernels.cu"
void lg_attn_prefill_out( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,float *__restrict__ __cuda_2,int __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7)
# 1543 "cuda/kernels.cu"
{__device_stub__Z19lg_attn_prefill_outPKfS0_Pfiiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 1592 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z9lg_linearPKfS0_S0_Pfiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6) {  const float *__T133;
 const float *__T134;
 const float *__T135;
 float *__T136;
__cudaLaunchPrologue(7); __T133 = __par0; __cudaSetupArgSimple(__T133, 0UL); __T134 = __par1; __cudaSetupArgSimple(__T134, 8UL); __T135 = __par2; __cudaSetupArgSimple(__T135, 16UL); __T136 = __par3; __cudaSetupArgSimple(__T136, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_linear))); }
# 1604 "cuda/kernels.cu"
void lg_linear( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6)
# 1608 "cuda/kernels.cu"
{__device_stub__Z9lg_linearPKfS0_S0_Pfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 1629 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z11lg_f32_gemmPKfS0_Pfiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  float *__restrict__ __par2,  int __par3,  int __par4,  int __par5) {  const float *__T137;
 const float *__T138;
 float *__T139;
__cudaLaunchPrologue(6); __T137 = __par0; __cudaSetupArgSimple(__T137, 0UL); __T138 = __par1; __cudaSetupArgSimple(__T138, 8UL); __T139 = __par2; __cudaSetupArgSimple(__T139, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaSetupArgSimple(__par5, 32UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_f32_gemm))); }
# 1638 "cuda/kernels.cu"
void lg_f32_gemm( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,float *__restrict__ __cuda_2,int __cuda_3,int __cuda_4,int __cuda_5)
# 1641 "cuda/kernels.cu"
{__device_stub__Z11lg_f32_gemmPKfS0_Pfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 1650 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z17lg_f32_gemm_tiledPKfS0_Pfiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  float *__restrict__ __par2,  int __par3,  int __par4,  int __par5) {  const float *__T140;
 const float *__T141;
 float *__T142;
__cudaLaunchPrologue(6); __T140 = __par0; __cudaSetupArgSimple(__T140, 0UL); __T141 = __par1; __cudaSetupArgSimple(__T141, 8UL); __T142 = __par2; __cudaSetupArgSimple(__T142, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaSetupArgSimple(__par5, 32UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_f32_gemm_tiled))); }
# 1658 "cuda/kernels.cu"
void lg_f32_gemm_tiled( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,float *__restrict__ __cuda_2,int __cuda_3,int __cuda_4,int __cuda_5)
# 1661 "cuda/kernels.cu"
{__device_stub__Z17lg_f32_gemm_tiledPKfS0_Pfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 1725 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z16lg_quantize_q8_0PKfPaPfii( const float *__restrict__ __par0,  int8_t *__restrict__ __par1,  float *__restrict__ __par2,  int __par3,  int __par4) {  const float *__T143;
 int8_t *__T144;
 float *__T145;
__cudaLaunchPrologue(5); __T143 = __par0; __cudaSetupArgSimple(__T143, 0UL); __T144 = __par1; __cudaSetupArgSimple(__T144, 8UL); __T145 = __par2; __cudaSetupArgSimple(__T145, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, int8_t *__restrict__, float *__restrict__, int, int))lg_quantize_q8_0))); }
# 1747 "cuda/kernels.cu"
void lg_quantize_q8_0( const float *__restrict__ __cuda_0,int8_t *__restrict__ __cuda_1,float *__restrict__ __cuda_2,int __cuda_3,int __cuda_4)
# 1750 "cuda/kernels.cu"
{__device_stub__Z16lg_quantize_q8_0PKfPaPfii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 1767 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z17lg_q8_0_gemm_dp4aPKhPKaPKfPfiii( const uint8_t *__restrict__ __par0,  const int8_t *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6) {  const uint8_t *__T146;
 const int8_t *__T147;
 const float *__T148;
 float *__T149;
__cudaLaunchPrologue(7); __T146 = __par0; __cudaSetupArgSimple(__T146, 0UL); __T147 = __par1; __cudaSetupArgSimple(__T147, 8UL); __T148 = __par2; __cudaSetupArgSimple(__T148, 16UL); __T149 = __par3; __cudaSetupArgSimple(__T149, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaLaunch(((char *)((void ( *)(const uint8_t *__restrict__, const int8_t *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_q8_0_gemm_dp4a))); }
# 1774 "cuda/kernels.cu"
void lg_q8_0_gemm_dp4a( const uint8_t *__restrict__ __cuda_0,const int8_t *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6)
# 1778 "cuda/kernels.cu"
{__device_stub__Z17lg_q8_0_gemm_dp4aPKhPKaPKfPfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 1868 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z20lg_q8_0_gemm_alignedPKhPKfPfiii( const uint8_t *__restrict__ __par0,  const float *__restrict__ __par1,  float *__restrict__ __par2,  int __par3,  int __par4,  int __par5) {  const uint8_t *__T150;
 const float *__T151;
 float *__T152;
__cudaLaunchPrologue(6); __T150 = __par0; __cudaSetupArgSimple(__T150, 0UL); __T151 = __par1; __cudaSetupArgSimple(__T151, 8UL); __T152 = __par2; __cudaSetupArgSimple(__T152, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaSetupArgSimple(__par5, 32UL); __cudaLaunch(((char *)((void ( *)(const uint8_t *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_q8_0_gemm_aligned))); }
# 1873 "cuda/kernels.cu"
void lg_q8_0_gemm_aligned( const uint8_t *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,float *__restrict__ __cuda_2,int __cuda_3,int __cuda_4,int __cuda_5)
# 1876 "cuda/kernels.cu"
{__device_stub__Z20lg_q8_0_gemm_alignedPKhPKfPfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 1894 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z12lg_q8_0_gemvPKhPKaPKfPfiiiS4_( const uint8_t *__restrict__ __par0,  const int8_t *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  const float *__restrict__ __par7) {  const uint8_t *__T153;
 const int8_t *__T154;
 const float *__T155;
 float *__T156;
 const float *__T157;
__cudaLaunchPrologue(8); __T153 = __par0; __cudaSetupArgSimple(__T153, 0UL); __T154 = __par1; __cudaSetupArgSimple(__T154, 8UL); __T155 = __par2; __cudaSetupArgSimple(__T155, 16UL); __T156 = __par3; __cudaSetupArgSimple(__T156, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __T157 = __par7; __cudaSetupArgSimple(__T157, 48UL); __cudaLaunch(((char *)((void ( *)(const uint8_t *__restrict__, const int8_t *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, const float *__restrict__))lg_q8_0_gemv))); }
# 1905 "cuda/kernels.cu"
void lg_q8_0_gemv( const uint8_t *__restrict__ __cuda_0,const int8_t *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,const float *__restrict__ __cuda_7)
# 1909 "cuda/kernels.cu"
{__device_stub__Z12lg_q8_0_gemvPKhPKaPKfPfiiiS4_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 1944 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z8lg_lreluPKfPffl( const float *__restrict__ __par0,  float *__restrict__ __par1,  float __par2,  long __par3) {  const float *__T158;
 float *__T159;
__cudaLaunchPrologue(4); __T158 = __par0; __cudaSetupArgSimple(__T158, 0UL); __T159 = __par1; __cudaSetupArgSimple(__T159, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, float, long))lg_lrelu))); }
# 1952 "cuda/kernels.cu"
void lg_lrelu( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,float __cuda_2,long __cuda_3)
# 1954 "cuda/kernels.cu"
{__device_stub__Z8lg_lreluPKfPffl( __cuda_0,__cuda_1,__cuda_2,__cuda_3);




}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z13lg_add_scaledPKfS0_Pflf( const float *__restrict__ __par0,  const float *__restrict__ __par1,  float *__restrict__ __par2,  long __par3,  float __par4) {  const float *__T160;
 const float *__T161;
 float *__T162;
__cudaLaunchPrologue(5); __T160 = __par0; __cudaSetupArgSimple(__T160, 0UL); __T161 = __par1; __cudaSetupArgSimple(__T161, 8UL); __T162 = __par2; __cudaSetupArgSimple(__T162, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, long, float))lg_add_scaled))); }
# 1964 "cuda/kernels.cu"
void lg_add_scaled( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,float *__restrict__ __cuda_2,long __cuda_3,float __cuda_4)
# 1967 "cuda/kernels.cu"
{__device_stub__Z13lg_add_scaledPKfS0_Pflf( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);



}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z21lg_upsample2x_nearestPKfPfiii( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2,  int __par3,  int __par4) {  const float *__T163;
 float *__T164;
__cudaLaunchPrologue(5); __T163 = __par0; __cudaSetupArgSimple(__T163, 0UL); __T164 = __par1; __cudaSetupArgSimple(__T164, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int, int, int))lg_upsample2x_nearest))); }
# 1980 "cuda/kernels.cu"
void lg_upsample2x_nearest( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2,int __cuda_3,int __cuda_4)
# 1983 "cuda/kernels.cu"
{__device_stub__Z21lg_upsample2x_nearestPKfPfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 1994 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z19lg_pixel_unshuffle2PKfPfiii( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2,  int __par3,  int __par4) {  const float *__T165;
 float *__T166;
__cudaLaunchPrologue(5); __T165 = __par0; __cudaSetupArgSimple(__T165, 0UL); __T166 = __par1; __cudaSetupArgSimple(__T166, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int, int, int))lg_pixel_unshuffle2))); }
# 2003 "cuda/kernels.cu"
void lg_pixel_unshuffle2( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2,int __cuda_3,int __cuda_4)
# 2006 "cuda/kernels.cu"
{__device_stub__Z19lg_pixel_unshuffle2PKfPfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 2022 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z16lg_pixel_shufflePKfPfiiii( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2,  int __par3,  int __par4,  int __par5) {  const float *__T167;
 float *__T168;
__cudaLaunchPrologue(6); __T167 = __par0; __cudaSetupArgSimple(__T167, 0UL); __T168 = __par1; __cudaSetupArgSimple(__T168, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 28UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int, int, int, int))lg_pixel_shuffle))); }
# 2052 "cuda/kernels.cu"
void lg_pixel_shuffle( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2,int __cuda_3,int __cuda_4,int __cuda_5)
# 2055 "cuda/kernels.cu"
{__device_stub__Z16lg_pixel_shufflePKfPfiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 2071 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z11lg_fft2_r2cPKfPfiiif( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2,  int __par3,  int __par4,  float __par5) {  const float *__T186;
 float *__T187;
__cudaLaunchPrologue(6); __T186 = __par0; __cudaSetupArgSimple(__T186, 0UL); __T187 = __par1; __cudaSetupArgSimple(__T187, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 28UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int, int, int, float))lg_fft2_r2c))); }
# 2172 "cuda/kernels.cu"
void lg_fft2_r2c( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2,int __cuda_3,int __cuda_4,float __cuda_5)
# 2175 "cuda/kernels.cu"
{__device_stub__Z11lg_fft2_r2cPKfPfiiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 2220 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z11lg_fft2_c2rPKfPfiiif( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2,  int __par3,  int __par4,  float __par5) {  const float *__T188;
 float *__T189;
__cudaLaunchPrologue(6); __T188 = __par0; __cudaSetupArgSimple(__T188, 0UL); __T189 = __par1; __cudaSetupArgSimple(__T189, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 28UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int, int, int, float))lg_fft2_c2r))); }
# 2227 "cuda/kernels.cu"
void lg_fft2_c2r( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2,int __cuda_3,int __cuda_4,float __cuda_5)
# 2230 "cuda/kernels.cu"
{__device_stub__Z11lg_fft2_c2rPKfPfiiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 2286 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z12lg_linear_rbPKfS0_S0_Pfiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6) {  const float *__T190;
 const float *__T191;
 const float *__T192;
 float *__T193;
__cudaLaunchPrologue(7); __T190 = __par0; __cudaSetupArgSimple(__T190, 0UL); __T191 = __par1; __cudaSetupArgSimple(__T191, 8UL); __T192 = __par2; __cudaSetupArgSimple(__T192, 16UL); __T193 = __par3; __cudaSetupArgSimple(__T193, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_linear_rb))); }
# 2426 "cuda/kernels.cu"
void lg_linear_rb( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6)
# 2430 "cuda/kernels.cu"
{__device_stub__Z12lg_linear_rbPKfS0_S0_Pfiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);

}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z13lg_conv1x1_rbPKfS0_S0_Pfiiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7) {  const float *__T194;
 const float *__T195;
 const float *__T196;
 float *__T197;
__cudaLaunchPrologue(8); __T194 = __par0; __cudaSetupArgSimple(__T194, 0UL); __T195 = __par1; __cudaSetupArgSimple(__T195, 8UL); __T196 = __par2; __cudaSetupArgSimple(__T196, 16UL); __T197 = __par3; __cudaSetupArgSimple(__T197, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv1x1_rb))); }
# 2438 "cuda/kernels.cu"
void lg_conv1x1_rb( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7)
# 2442 "cuda/kernels.cu"
{__device_stub__Z13lg_conv1x1_rbPKfS0_S0_Pfiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);

}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z18lg_layer_norm_warpPKfS0_S0_Pfiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  float __par6) {  const float *__T200;
 const float *__T201;
 const float *__T202;
 float *__T203;
__cudaLaunchPrologue(7); __T200 = __par0; __cudaSetupArgSimple(__T200, 0UL); __T201 = __par1; __cudaSetupArgSimple(__T201, 8UL); __T202 = __par2; __cudaSetupArgSimple(__T202, 16UL); __T203 = __par3; __cudaSetupArgSimple(__T203, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float))lg_layer_norm_warp))); }
# 2470 "cuda/kernels.cu"
void lg_layer_norm_warp( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,float __cuda_6)
# 2473 "cuda/kernels.cu"
{__device_stub__Z18lg_layer_norm_warpPKfS0_S0_Pfiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 2516 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z15lg_conv3x3_tilePKfS0_S0_Pfiiiiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7,  int __par8,  float __par9) {  const float *__T204;
 const float *__T205;
 const float *__T206;
 float *__T207;
__cudaLaunchPrologue(10); __T204 = __par0; __cudaSetupArgSimple(__T204, 0UL); __T205 = __par1; __cudaSetupArgSimple(__T205, 8UL); __T206 = __par2; __cudaSetupArgSimple(__T206, 16UL); __T207 = __par3; __cudaSetupArgSimple(__T207, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaSetupArgSimple(__par8, 48UL); __cudaSetupArgSimple(__par9, 52UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_conv3x3_tile))); }
# 2609 "cuda/kernels.cu"
void lg_conv3x3_tile( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,float __cuda_9)
# 2613 "cuda/kernels.cu"
{__device_stub__Z15lg_conv3x3_tilePKfS0_S0_Pfiiiiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9);
# 2710 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z15lg_conv1x1_tilePKfS0_S0_Pfiiiiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7,  int __par8,  float __par9) {  const float *__T208;
 const float *__T209;
 const float *__T210;
 float *__T211;
__cudaLaunchPrologue(10); __T208 = __par0; __cudaSetupArgSimple(__T208, 0UL); __T209 = __par1; __cudaSetupArgSimple(__T209, 8UL); __T210 = __par2; __cudaSetupArgSimple(__T210, 16UL); __T211 = __par3; __cudaSetupArgSimple(__T211, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaSetupArgSimple(__par8, 48UL); __cudaSetupArgSimple(__par9, 52UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_conv1x1_tile))); }
# 2729 "cuda/kernels.cu"
void lg_conv1x1_tile( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,float __cuda_9)
# 2733 "cuda/kernels.cu"
{__device_stub__Z15lg_conv1x1_tilePKfS0_S0_Pfiiiiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9);
# 2807 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z12lg_conv2x2s2PKfS0_S0_Pfiiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7) {  const float *__T212;
 const float *__T213;
 const float *__T214;
 float *__T215;
__cudaLaunchPrologue(8); __T212 = __par0; __cudaSetupArgSimple(__T212, 0UL); __T213 = __par1; __cudaSetupArgSimple(__T213, 8UL); __T214 = __par2; __cudaSetupArgSimple(__T214, 16UL); __T215 = __par3; __cudaSetupArgSimple(__T215, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv2x2s2))); }
# 2828 "cuda/kernels.cu"
void lg_conv2x2s2( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7)
# 2832 "cuda/kernels.cu"
{__device_stub__Z12lg_conv2x2s2PKfS0_S0_Pfiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 2876 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z12lg_conv_t2x2PKfS0_S0_Pfiiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7) {  const float *__T216;
 const float *__T217;
 const float *__T218;
 float *__T219;
__cudaLaunchPrologue(8); __T216 = __par0; __cudaSetupArgSimple(__T216, 0UL); __T217 = __par1; __cudaSetupArgSimple(__T217, 8UL); __T218 = __par2; __cudaSetupArgSimple(__T218, 16UL); __T219 = __par3; __cudaSetupArgSimple(__T219, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv_t2x2))); }
# 2884 "cuda/kernels.cu"
void lg_conv_t2x2( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7)
# 2888 "cuda/kernels.cu"
{__device_stub__Z12lg_conv_t2x2PKfS0_S0_Pfiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 2925 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z16lg_window_gatherPKfPfiiiiiiiii( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2,  int __par3,  int __par4,  int __par5,  int __par6,  int __par7,  int __par8,  int __par9,  int __par10) {  const float *__T220;
 float *__T221;
__cudaLaunchPrologue(11); __T220 = __par0; __cudaSetupArgSimple(__T220, 0UL); __T221 = __par1; __cudaSetupArgSimple(__T221, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 28UL); __cudaSetupArgSimple(__par6, 32UL); __cudaSetupArgSimple(__par7, 36UL); __cudaSetupArgSimple(__par8, 40UL); __cudaSetupArgSimple(__par9, 44UL); __cudaSetupArgSimple(__par10, 48UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int, int, int, int, int, int, int, int, int))lg_window_gather))); }
# 2955 "cuda/kernels.cu"
void lg_window_gather( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2,int __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,int __cuda_9,int __cuda_10)
# 2958 "cuda/kernels.cu"
{__device_stub__Z16lg_window_gatherPKfPfiiiiiiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10);
# 2968 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z17lg_window_scatterPKfPfiiiiiiiii( const float *__restrict__ __par0,  float *__restrict__ __par1,  int __par2,  int __par3,  int __par4,  int __par5,  int __par6,  int __par7,  int __par8,  int __par9,  int __par10) {  const float *__T222;
 float *__T223;
__cudaLaunchPrologue(11); __T222 = __par0; __cudaSetupArgSimple(__T222, 0UL); __T223 = __par1; __cudaSetupArgSimple(__T223, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 28UL); __cudaSetupArgSimple(__par6, 32UL); __cudaSetupArgSimple(__par7, 36UL); __cudaSetupArgSimple(__par8, 40UL); __cudaSetupArgSimple(__par9, 44UL); __cudaSetupArgSimple(__par10, 48UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, float *__restrict__, int, int, int, int, int, int, int, int, int))lg_window_scatter))); }
# 2972 "cuda/kernels.cu"
void lg_window_scatter( const float *__restrict__ __cuda_0,float *__restrict__ __cuda_1,int __cuda_2,int __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,int __cuda_9,int __cuda_10)
# 2975 "cuda/kernels.cu"
{__device_stub__Z17lg_window_scatterPKfPfiiiiiiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10);
# 2985 "cuda/kernels.cu"
}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z13lg_conv3x3_q2PKfS0_S0_Pfiiiiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7,  int __par8,  float __par9) {  const float *__T224;
 const float *__T225;
 const float *__T226;
 float *__T227;
__cudaLaunchPrologue(10); __T224 = __par0; __cudaSetupArgSimple(__T224, 0UL); __T225 = __par1; __cudaSetupArgSimple(__T225, 8UL); __T226 = __par2; __cudaSetupArgSimple(__T226, 16UL); __T227 = __par3; __cudaSetupArgSimple(__T227, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaSetupArgSimple(__par8, 48UL); __cudaSetupArgSimple(__par9, 52UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_conv3x3_q2))); }
# 3313 "cuda/kernels.cu"
void lg_conv3x3_q2( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,float __cuda_9)
# 3316 "cuda/kernels.cu"
{__device_stub__Z13lg_conv3x3_q2PKfS0_S0_Pfiiiiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9);

}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z16lg_conv3x3_q2ng2PKfS0_S0_Pfiiiiif( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7,  int __par8,  float __par9) {  const float *__T228;
 const float *__T229;
 const float *__T230;
 float *__T231;
__cudaLaunchPrologue(10); __T228 = __par0; __cudaSetupArgSimple(__T228, 0UL); __T229 = __par1; __cudaSetupArgSimple(__T229, 8UL); __T230 = __par2; __cudaSetupArgSimple(__T230, 16UL); __T231 = __par3; __cudaSetupArgSimple(__T231, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaSetupArgSimple(__par8, 48UL); __cudaSetupArgSimple(__par9, 52UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_conv3x3_q2ng2))); }
# 3321 "cuda/kernels.cu"
void lg_conv3x3_q2ng2( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,float __cuda_9)
# 3324 "cuda/kernels.cu"
{__device_stub__Z16lg_conv3x3_q2ng2PKfS0_S0_Pfiiiiif( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9);

}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z16lg_conv3x3_catq0PKfS0_S0_PfiiiiifS0_S0_S0_S0_S0_iiiiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7,  int __par8,  float __par9,  const float *__restrict__ __par10,  const float *__restrict__ __par11,  const float *__restrict__ __par12,  const float *__restrict__ __par13,  const float *__restrict__ __par14,  int __par15,  int __par16,  int __par17,  int __par18,  int __par19,  int __par20) {  const float *__T232;
 const float *__T233;
 const float *__T234;
 float *__T235;
 const float *__T236;
 const float *__T237;
 const float *__T238;
 const float *__T239;
 const float *__T240;
__cudaLaunchPrologue(21); __T232 = __par0; __cudaSetupArgSimple(__T232, 0UL); __T233 = __par1; __cudaSetupArgSimple(__T233, 8UL); __T234 = __par2; __cudaSetupArgSimple(__T234, 16UL); __T235 = __par3; __cudaSetupArgSimple(__T235, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaSetupArgSimple(__par8, 48UL); __cudaSetupArgSimple(__par9, 52UL); __T236 = __par10; __cudaSetupArgSimple(__T236, 56UL); __T237 = __par11; __cudaSetupArgSimple(__T237, 64UL); __T238 = __par12; __cudaSetupArgSimple(__T238, 72UL); __T239 = __par13; __cudaSetupArgSimple(__T239, 80UL); __T240 = __par14; __cudaSetupArgSimple(__T240, 88UL); __cudaSetupArgSimple(__par15, 96UL); __cudaSetupArgSimple(__par16, 100UL); __cudaSetupArgSimple(__par17, 104UL); __cudaSetupArgSimple(__par18, 108UL); __cudaSetupArgSimple(__par19, 112UL); __cudaSetupArgSimple(__par20, 116UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, int, int, int, int, int, int))lg_conv3x3_catq0))); }
# 3329 "cuda/kernels.cu"
void lg_conv3x3_catq0( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,float __cuda_9,const float *__restrict__ __cuda_10,const float *__restrict__ __cuda_11,const float *__restrict__ __cuda_12,const float *__restrict__ __cuda_13,const float *__restrict__ __cuda_14,int __cuda_15,int __cuda_16,int __cuda_17,int __cuda_18,int __cuda_19,int __cuda_20)
# 3336 "cuda/kernels.cu"
{__device_stub__Z16lg_conv3x3_catq0PKfS0_S0_PfiiiiifS0_S0_S0_S0_S0_iiiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10,__cuda_11,__cuda_12,__cuda_13,__cuda_14,__cuda_15,__cuda_16,__cuda_17,__cuda_18,__cuda_19,__cuda_20);


}
# 1 "kernels.cudafe1.stub.c"
void __device_stub__Z19lg_conv3x3_catq0ng2PKfS0_S0_PfiiiiifS0_S0_S0_S0_S0_iiiiii( const float *__restrict__ __par0,  const float *__restrict__ __par1,  const float *__restrict__ __par2,  float *__restrict__ __par3,  int __par4,  int __par5,  int __par6,  int __par7,  int __par8,  float __par9,  const float *__restrict__ __par10,  const float *__restrict__ __par11,  const float *__restrict__ __par12,  const float *__restrict__ __par13,  const float *__restrict__ __par14,  int __par15,  int __par16,  int __par17,  int __par18,  int __par19,  int __par20) {  const float *__T241;
 const float *__T242;
 const float *__T243;
 float *__T244;
 const float *__T245;
 const float *__T246;
 const float *__T247;
 const float *__T248;
 const float *__T249;
__cudaLaunchPrologue(21); __T241 = __par0; __cudaSetupArgSimple(__T241, 0UL); __T242 = __par1; __cudaSetupArgSimple(__T242, 8UL); __T243 = __par2; __cudaSetupArgSimple(__T243, 16UL); __T244 = __par3; __cudaSetupArgSimple(__T244, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaSetupArgSimple(__par7, 44UL); __cudaSetupArgSimple(__par8, 48UL); __cudaSetupArgSimple(__par9, 52UL); __T245 = __par10; __cudaSetupArgSimple(__T245, 56UL); __T246 = __par11; __cudaSetupArgSimple(__T246, 64UL); __T247 = __par12; __cudaSetupArgSimple(__T247, 72UL); __T248 = __par13; __cudaSetupArgSimple(__T248, 80UL); __T249 = __par14; __cudaSetupArgSimple(__T249, 88UL); __cudaSetupArgSimple(__par15, 96UL); __cudaSetupArgSimple(__par16, 100UL); __cudaSetupArgSimple(__par17, 104UL); __cudaSetupArgSimple(__par18, 108UL); __cudaSetupArgSimple(__par19, 112UL); __cudaSetupArgSimple(__par20, 116UL); __cudaLaunch(((char *)((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, int, int, int, int, int, int))lg_conv3x3_catq0ng2))); }
# 3342 "cuda/kernels.cu"
void lg_conv3x3_catq0ng2( const float *__restrict__ __cuda_0,const float *__restrict__ __cuda_1,const float *__restrict__ __cuda_2,float *__restrict__ __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7,int __cuda_8,float __cuda_9,const float *__restrict__ __cuda_10,const float *__restrict__ __cuda_11,const float *__restrict__ __cuda_12,const float *__restrict__ __cuda_13,const float *__restrict__ __cuda_14,int __cuda_15,int __cuda_16,int __cuda_17,int __cuda_18,int __cuda_19,int __cuda_20)
# 3349 "cuda/kernels.cu"
{__device_stub__Z19lg_conv3x3_catq0ng2PKfS0_S0_PfiiiiifS0_S0_S0_S0_S0_iiiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10,__cuda_11,__cuda_12,__cuda_13,__cuda_14,__cuda_15,__cuda_16,__cuda_17,__cuda_18,__cuda_19,__cuda_20);


}
# 1 "kernels.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T551) {  __nv_dummy_param_ref(__T551); __nv_save_fatbinhandle_for_managed_rt(__T551); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, int, int, int, int, int, int))lg_conv3x3_catq0ng2), lg_conv3x3_catq0ng2, 256); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, int, int, int, int, int, int))lg_conv3x3_catq0), lg_conv3x3_catq0, 128); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_conv3x3_q2ng2), lg_conv3x3_q2ng2, 256); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_conv3x3_q2), lg_conv3x3_q2, 128); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int, int, int, int, int, int, int, int, int))lg_window_scatter), lg_window_scatter, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int, int, int, int, int, int, int, int, int))lg_window_gather), lg_window_gather, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv_t2x2), lg_conv_t2x2, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv2x2s2), lg_conv2x2s2, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_conv1x1_tile), lg_conv1x1_tile, 128); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_conv3x3_tile), lg_conv3x3_tile, 256); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float))lg_layer_norm_warp), lg_layer_norm_warp, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv1x1_rb), lg_conv1x1_rb, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_linear_rb), lg_linear_rb, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int, int, int, float))lg_fft2_c2r), lg_fft2_c2r, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int, int, int, float))lg_fft2_r2c), lg_fft2_r2c, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int, int, int, int))lg_pixel_shuffle), lg_pixel_shuffle, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int, int, int))lg_pixel_unshuffle2), lg_pixel_unshuffle2, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int, int, int))lg_upsample2x_nearest), lg_upsample2x_nearest, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, long, float))lg_add_scaled), lg_add_scaled, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, float, long))lg_lrelu), lg_lrelu, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const uint8_t *__restrict__, const int8_t *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, const float *__restrict__))lg_q8_0_gemv), lg_q8_0_gemv, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const uint8_t *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_q8_0_gemm_aligned), lg_q8_0_gemm_aligned, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const uint8_t *__restrict__, const int8_t *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_q8_0_gemm_dp4a), lg_q8_0_gemm_dp4a, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, int8_t *__restrict__, float *__restrict__, int, int))lg_quantize_q8_0), lg_quantize_q8_0, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_f32_gemm_tiled), lg_f32_gemm_tiled, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_f32_gemm), lg_f32_gemm, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int))lg_linear), lg_linear, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int))lg_attn_prefill_out), lg_attn_prefill_out, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(float *__restrict__, int, int))lg_attn_prefill_softmax), lg_attn_prefill_softmax, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_attn_prefill_fused), lg_attn_prefill_fused, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_attn_prefill_scores), lg_attn_prefill_scores, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, float, int))lg_attn_flash), lg_attn_flash, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, float))lg_attn_gqa), lg_attn_gqa, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(int *__restrict__, int))lg_set_i32), lg_set_i32, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const struct float4 *__restrict__, struct float4 *__restrict__, int))lg_copy_row), lg_copy_row, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const uint8_t *__restrict__, int, float *__restrict__, int))lg_get_row_q8_0_aligned), lg_get_row_q8_0_aligned, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(float *__restrict__, const float *__restrict__, int, int, int))lg_merge_2x2), lg_merge_2x2, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(float *__restrict__, const float *__restrict__, int, int, int))lg_extract_rows), lg_extract_rows, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int, int, int, float))lg_conv3x3_winograd), lg_conv3x3_winograd, 256); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int))lg_linear_1x1), lg_linear_1x1, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int, int))lg_conv_kxk), lg_conv_kxk, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv4x4s4), lg_conv4x4s4, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv3x3s1p1), lg_conv3x3s1p1, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, int, int))lg_conv1x1), lg_conv1x1, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, int *__restrict__, float *__restrict__, int, int))lg_argmax), lg_argmax, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int, int))lg_channel_mean), lg_channel_mean, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, long))lg_copy), lg_copy, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int))lg_channel_scale), lg_channel_scale, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, const float *__restrict__, const float *__restrict__, int, int))lg_channel_affine), lg_channel_affine, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float))lg_channel_layer_norm), lg_channel_layer_norm, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(float *__restrict__, const float *__restrict__, const float *__restrict__, int, int))lg_row_affine), lg_row_affine, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, float, int))lg_scale), lg_scale, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(float *__restrict__, const float *__restrict__, int))lg_add_inplace), lg_add_inplace, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int))lg_mul), lg_mul, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int))lg_add), lg_add, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int))lg_sigmoid), lg_sigmoid, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int))lg_relu), lg_relu, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int))lg_gelu_erf), lg_gelu_erf, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, float *__restrict__, int))lg_gelu_tanh), lg_gelu_tanh, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(float *__restrict__, const float *__restrict__, int))lg_silu_mul), lg_silu_mul, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(float *__restrict__, const float *__restrict__, const float *__restrict__, int, int, int))lg_rope_2d), lg_rope_2d, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(float *__restrict__, const int *__restrict__, int, int, int, float))lg_rope_neox), lg_rope_neox, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float))lg_layer_norm_2pass), lg_layer_norm_2pass, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float))lg_layer_norm), lg_layer_norm, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(const float *__restrict__, const float *__restrict__, float *__restrict__, int, int, float))lg_rms_norm), lg_rms_norm, (-1)); __cudaRegisterEntry(__T551, ((void ( *)(void))lg_noop), lg_noop, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
