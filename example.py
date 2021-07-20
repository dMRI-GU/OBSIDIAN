import numpy as np
import algorithm as ra
import copy

para_sim_def={
  #Default simulation parameters
  'sig': 1., #sigma of gaussian noise
  'b': np.linspace(0,3000,21),
  'SNR': 10,
  'sim_para': [2.2,0.4,.8], #diff coefficient water [m^2/s*1e9]
  #'num_repeat': 1.,
  'num_repeat': 50,
  'guess': None,
  'bound': None,
  'model_func_name': 'biexp',
  'sigma_av_corr': 2.3,
}

bias_corr=ra.RicianBiasCorr(debug=False)

bmax=3000
num_b=21
SNR_list=[50]

for i_snr,SNR in enumerate(SNR_list):
  print(SNR)
  #RICE from generated data
  #generate rice signal
  para_sim=copy.deepcopy(para_sim_def)
  para_sim['b']=np.linspace(0,bmax,num_b)
  para_sim['sim_para']=[2.2,0.4,.8]
  para_sim['SNR']=SNR
  print('Generate model data')
  num_repeat=para_sim['num_repeat']
  model_func=bias_corr.func_dict['biexp']
  in_para=para_sim['sim_para']+[SNR]
  b=para_sim['b']
  sig=para_sim['sig']
  sig_input=model_func(b,*in_para)

  #biexp 1D
  print('Biexp 1D')
  sig_input_all=np.tile(sig_input,num_repeat*10)
  rice_signal=bias_corr.gen_rice_signal(sig_input_all,
                                        sig).reshape(num_repeat*10,len(b))
  #set model func
  para_sim['model_func_name']='biexp'
  para_sim['guess']=(2.,.5,.5,SNR*.8)
  para_sim['bound']=([0.,0.,0.1,0.],[4.,1.,.9,np.inf])
  para_sim['model_func_name']='biexp'

  res=bias_corr.run_bias_corr(para_sim,rice_signal,make_output=False)
  res_save=res[0][np.arange(num_repeat),res[2]]
  print(np.mean(res_save,axis=0))

  #biexp 1D fixed sigma
  print('Biexp 1D fixed sigma')
  sig_input_all=np.tile(sig_input,num_repeat*10)
  rice_signal=bias_corr.gen_rice_signal(sig_input_all,
                                        sig).reshape(num_repeat*10,len(b))
  #set model func
  para_sim['model_func_name']='biexp'
  para_sim['guess']=(2.,.5,.5,SNR*.8)
  para_sim['bound']=([0.,0.,0.1,0.],[4.,1.,.9,np.inf])
  para_sim['model_func_name']='biexp'

  res=bias_corr.run_bias_corr(para_sim,rice_signal,make_output=False,sigma=1)
  res_save=res[0][np.arange(num_repeat),res[2]]
  print(np.mean(res_save,axis=0))


  #biexp3D
  print('Biexp 3D')
  sig_input_all=np.tile(sig_input,3*num_repeat)
  rice_signal=bias_corr.gen_rice_signal(sig_input_all,
                                        sig).reshape(num_repeat,3*len(b))
  #set model func
  #algorithm without weight
  para_sim['b']=np.tile(np.linspace(0,bmax,num_b),3).reshape(3,num_b)
  para_sim['guess']=(2.,.5,.5,2.,.5,.5,2.,.5,.5,SNR*.8)
  para_sim['bound']=([0.,0.,0.1,0.,0.,0.1,0.,0.,0.1,0.],
                     [4.,1.,.9,4.,1.,.9,4.,1.,.9,np.inf])
  para_sim['model_func_name']='biexp_3D'
  para_sim['sigma_av_corr']=5.4
  model_func=bias_corr.func_dict[para_sim['model_func_name']]
  res=bias_corr.run_bias_corr(para_sim,rice_signal,
                              make_output=False,
                              num_par=None)
  res_save=res[0][np.arange(num_repeat),res[2]]
  print(np.mean(res_save,axis=0))

  #biexp3D composite
  print('Biexp 3D composite')
  num_comp=100
  sig_input_all=np.tile(sig_input,3*num_comp*num_repeat)
  rice_signal=bias_corr.gen_rice_signal(sig_input_all,
                                        sig).reshape(num_repeat,3*len(b)*num_comp)
  #set model func
  #algorithm without weight
  para_sim['b']=(np.tile(np.linspace(0,bmax,num_b),3*num_comp).reshape(3,num_b*num_comp))
  para_sim['guess']=(2.,.5,.5,2.,.5,.5,2.,.5,.5,SNR*.8)
  para_sim['bound']=([0.,0.,0.1,0.,0.,0.1,0.,0.,0.1,0.],
                     [4.,1.,.9,4.,1.,.9,4.,1.,.9,np.inf])
  para_sim['model_func_name']='biexp_3D'
  model_func=bias_corr.func_dict[para_sim['model_func_name']]
  
  res=bias_corr.run_bias_corr(para_sim,rice_signal,
                              make_output=False,
                              num_par=None)
  res_save=res[0][np.arange(num_repeat),res[2]]
  print(np.mean(res_save,axis=0))


  #biexp3D average
  print('Biexp 3D average')
  num_avrg=100
  sig_input_all=np.tile(sig_input,3*num_avrg*num_repeat)
  rice_signal=bias_corr.gen_rice_signal(sig_input_all,
                                        sig).reshape(num_avrg,num_repeat,3*len(b))
  rice_signal_avrg=np.mean(rice_signal,axis=0)
  #set model func
  #algorithm without weight
  para_sim['b']=np.tile(np.linspace(0,bmax,num_b),3).reshape(3,num_b)
  para_sim['guess']=(2.,.5,.5,2.,.5,.5,2.,.5,.5,SNR*.8)
  para_sim['bound']=([0.,0.,0.1,0.,0.,0.1,0.,0.,0.1,0.],
                     [4.,1.,.9,4.,1.,.9,4.,1.,.9,np.inf])
  para_sim['model_func_name']='biexp_3D'
  para_sim['N_avrg']=num_avrg
  para_sim['sigma_av_corr']=5.4
  model_func=bias_corr.func_dict[para_sim['model_func_name']]
  res=bias_corr.run_bias_corr(para_sim,rice_signal_avrg,
                              make_output=False,
                              num_par=None)
  res_save=res[0][np.arange(num_repeat),res[2]]
  print(np.mean(res_save,axis=0))





