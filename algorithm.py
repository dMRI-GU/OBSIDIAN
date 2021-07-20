# <codecell>
import numpy as np
import scipy.stats as st
import scipy.special as sp
import scipy.optimize as so
import time
import os
from imp import reload
import multiprocessing as mp

method_grad='trf' #'dogbox','lm',None

class RicianBiasCorr:
  """
  Class for Rician Bias correction
  """

  """
  Rican noise in diffusion data
  """

  def rice(self,x,nu,sig,cut_off=20,log_out=False):
    """
    Probability distribution function
    rice_scipy(x,nu/sig,scale=sig)==rice(x,nu,sig) (wikipedia)
    for asymptotic expression check Andersen 1996  (Letters to editor)
    cut_off: snr cut_off for asymptotic expression
    log_out: output log of probability
    """
    if np.isscalar(nu):
      is_scalar=True
      nu=np.array([nu])
      if np.isscalar(x):
        x=np.array([x])
    else:
      nu=np.array(nu)
      is_scalar=False
    x=np.array(x)
    res=np.zeros_like(nu,dtype=np.float)
    sig=np.ones_like(nu)*sig
    #assert(np.sum(sig==0)==0), "Sigma is zero; check bounds"
    snr=nu/sig
    mask_cut=(snr<cut_off)
    if not log_out:
      res[mask_cut]=st.rice.pdf(x[mask_cut],snr[mask_cut],scale=sig[mask_cut])
      res[~mask_cut]=(np.sqrt(x[~mask_cut]/nu[~mask_cut])*
                      np.sqrt(1/(2*np.pi*sig[~mask_cut]**2))*
                      np.exp(-(x[~mask_cut]-nu[~mask_cut])**2/
                             (2*sig[~mask_cut]**2)))
    else:
      x_cal=x[mask_cut]/sig[mask_cut]
      b_cal=snr[mask_cut]
      res[mask_cut]=(np.log(x_cal/sig[mask_cut])-
        (((x_cal-b_cal)**2)/2.)+np.log(sp.i0e(x_cal*b_cal)))
      res[~mask_cut]=(1/2.*np.log(x[~mask_cut]/(nu[~mask_cut]*
                                              2*np.pi*sig[~mask_cut]**2))-
                      (x[~mask_cut]-nu[~mask_cut])**2/
                      (2*sig[~mask_cut]**2))
    if is_scalar:
      res=res[0]
    return res

  def diff_mono(self,b,*p):
    """
    b: b-factor (=gamma^2*g^2*delta^2*(Delta-delta/3)) [s/mm^2]
    p[0] D: diffusion coefficient [m^2/s*1E-9]
    p[1] scale: S0 scaling of function
    """
    return p[1]*np.exp(-b*p[0]*1E-3)

  def diff_biexp(self,b,*p):
    """
    b: b-factor (=gamma^2*g^2*delta^2*(Delta-delta/3)) [s/mm^2]
    p[0] D1: diffusion coefficient 1[m^2/s*1E-9]
    p[1] D2: diffusion coefficient 2[m^2/s*1E-9]
    p[2] f1: proportion compartment with D1
    p[3] scale: S0 scaling of function
    """
    return p[3]*(p[2]*np.exp(-b*p[0]*1E-3)+(1-p[2])*np.exp(-b*p[1]*1E-3))

  def diff_gamma(self,b,*p):
    """
    b: b-factor (=gamma^2*g^2*delta^2*(Delta-delta/3)) [s/mm^2]
    p[0] k: exponent
    p[1] theta: theta
    p[2] scale: S0 scaling of function
    """
    return p[2]*np.power((1+p[1]*b*1e-3),-p[0])

  def diff_kurtosis(self,b,*p):
    """
    b: b-factor (=gamma^2*g^2*delta^2*(Delta-delta/3)) [s/mm^2]
    p[0] D: mean diffusion coefficient [m^2/s*1E-9]
    p[1] K: Kurtosis factor
    p[2] scale: S0 scaling of function
    """
    return p[2]*np.exp(-b*p[0]*1e-3+(b*p[0]*1E-3)**2*p[1]/6.)

  def diff_stretched(self,b,*p):
    """
    b: b-factor (=gamma^2*g^2*delta^2*(Delta-delta/3)) [s/mm^2]
    p[0] DDC: distributed diffusion coefficient
    p[1] alpha: stretching coefficient
    p[2] scale: S0 scaling of function
    """
    return p[2]*np.exp(-np.power(b*p[0]*1e-3,p[1]))

  def monoexp_3D(self,b,*p,no_ravel=False):
    """
    numpy version
    N dimensional monoexponential function
    b: b values; shape (N,#b-values)

    p[0],p[1],p[2]: diff coeff for 3 dimension (D1,D2,D3)
    p[3]: S0 shared by all dimensions
    """
    D=np.array([p[0],p[1],p[2]])[:,np.newaxis]
    gaussNd=p[3]*np.exp(-b*D*1E-3)
    if no_ravel:
      return gaussNd
    else:
      return np.ravel(gaussNd)

  def biexp_3D(self,b,*p,no_ravel=False):
    """
    3D monoexponential function
    b: b values; shape (3,#b-values)

    p[0],p[1],p[2]: diff coeff for 3 dimension (D1,D2,D3)
    p[3]: S0 shared by all dimensions
    """
    D1=np.array([p[0],p[3],p[6]])[:,np.newaxis]
    D2=np.array([p[1],p[4],p[7]])[:,np.newaxis]
    f=np.array([p[2],p[5],p[8]])[:,np.newaxis]
    diff3d=p[-1]*(f*np.exp(-b*D1*1E-3)+(1-f)*np.exp(-b*D2*1E-3))
    if no_ravel:
      return diff3d
    else:
      return np.ravel(diff3d)

  def kurtosis_3D(self,b,*p,no_ravel=False):
    """
    3D kurtosis function
    b: b values; shape (3,#b-values) [s/mm^2]
    unit D [m^2/s*1E-9]
    p[0],p[2],p[4]: D coeff for 3 dimension (D1,D2,D3)
    p[1],p[3],p[5]: K Kurtosis factor for 3 dimension (K1,K2,K3)
    p[6]: S0 shared by all dimensions
    """
    D=np.array([p[0],p[2],p[4]])[:,np.newaxis]
    K=np.array([p[1],p[3],p[5]])[:,np.newaxis]
    diff3d=p[-1]*np.exp(-b*D*1E-3+(b*D*1E-3)**2*K/6.)
    if no_ravel:
      return diff3d
    else:
      return np.ravel(diff3d)

  def gamma_3D(self,b,*p,no_ravel=False):
    """
    3D gamma diffusion function
    b: b values; shape (3,#b-values) [s/mm^2]
    unit D [m^2/s*1E-9]
    p[0],p[2],p[4]: exponent k 3D
    p[1],p[3],p[5]: theta 3D
    p[6]: S0 shared by all dimensions
    """
    k=np.array([p[0],p[2],p[4]])[:,np.newaxis]
    theta=np.array([p[1],p[3],p[5]])[:,np.newaxis]
    diff3d=p[-1]*np.power((1+theta*b*1e-3),-k)
    if no_ravel:
      return diff3d
    else:
      return np.ravel(diff3d)

  def stretched_3D(self,b,*p,no_ravel=False):
    """
    3D gamma diffusion function
    b: b values; shape (3,#b-values) [s/mm^2]
    unit D [m^2/s*1E-9]
    p[0],p[2],p[4]: DDC: distributed diffusion coefficient
    p[1],p[3],p[5]: alpha: stretching coefficient
    p[6]: S0 shared by all dimensions
    """
    DDC=np.array([p[0],p[2],p[4]])[:,np.newaxis]
    alpha=np.array([p[1],p[3],p[5]])[:,np.newaxis]
    diff3d=p[-1]*np.exp(-np.power(b*DDC*1e-3,alpha))
    if no_ravel:
      return diff3d
    else:
      return np.ravel(diff3d)

  def diff_stretched(self,b,*p):
    """
    b: b-factor (=gamma^2*g^2*delta^2*(Delta-delta/3)) [s/mm^2]
    p[0] DDC: distributed diffusion coefficient
    p[1] alpha: stretching coefficient
    p[2] scale: S0 scaling of function
    """
    return p[2]*np.exp(-np.power(b*p[0]*1e-3,p[1]))

  def test_lin(self,b,*p):
    """
    b: b-factor
    p[0]
    p[1]
    """
    return np.abs(p[1]*b*1e-3+p[0])

  def rice_rv(self,nu,sig,size=100,random_state=None):
    """
    Rician distributed random variable
    rice_scipy(x,nu/sig,scale=sig)==rice(x,nu,sig)
    nu: can be array but len(nu)==size
    """
    return st.rice.rvs(nu/sig,scale=sig,size=size,random_state=random_state)

  def rice_mean(self,nu,sig):
    """
    mean of rician distribution
    """
    return st.rice.mean(nu/sig,scale=sig)

  def rice_mean_high(self,nu,sig,cut_off=20):
    """
    mean of rician distribution
    use gaussian for high nu/sig values
    """
    res=np.zeros_like(nu,dtype=np.float)
    sig=np.ones_like(nu)*sig
    res[nu/sig<cut_off]=st.rice.mean(nu[nu/sig<cut_off]/sig[nu/sig<cut_off],
                                     scale=sig[nu/sig<cut_off])
    res[nu/sig>=cut_off]=np.sqrt(nu[nu/sig>=cut_off]**2+sig[nu/sig>=cut_off]**2)
    return res


  def rice_var(self,nu,sig):
    """
    variance of rician distribution
    """
    return st.rice.var(nu/sig,scale=sig)

  def rice_std(self,nu,sig):
    """
    variance of rician distribution
    doesn't work arrays for nu
    """
    return st.rice.std(nu/sig,scale=sig)

  def rice_std_arr(self,nu,sig):
    """
    variance of rician distribution
    std=(2*sig**2+nu**2-mean**2)**.5
    """
    return np.sqrt(2*sig**2+nu**2-st.rice.mean(nu/sig,scale=sig)**2)


  def rice_std_high(self,nu,sig,cut_off=20):
    """
    variance of rician distribution
    use gaussian for high nu/sig values
    """
    res=np.zeros_like(nu)
    res[nu/sig<cut_off]=self.rice_std_arr(nu[nu/sig<cut_off],sig)
    res[nu/sig>=cut_off]=np.ones_like(nu[nu/sig>=cut_off])*sig
    return res

  def alpha_func(self,SNR,N=1):
    """
    residual function alpha
    N=1: Use approximation function from paper
    Equation in the N>1 works well for N>=4
    """
    if N==1:
      val=np.sqrt(np.pi/2.)*np.exp(-4./3.*SNR)+np.sqrt(2./np.pi)*sp.erf(.5*SNR)
    else:
      eta=np.sqrt(N)*(self.rice_mean_high(SNR,1)-SNR)
      val=1/np.sqrt(N)*(np.sqrt(2/np.pi)*np.exp(-eta**2/2.)+eta*sp.erf(eta/np.sqrt(2)))
    return val

  def __init__(self,debug=False):
    """
    init function
    """
    self.debug=debug

    self.default_para_dict={
      #Default simulation parameters
      'sig': 1, #sigma of gaussian noise
      'b': np.linspace(0,3000,21),
      'SNR': 10,
      'sim_para': [2.3], #diff coefficient water [m^2/s*1E-9]
      #'num_repeat': 1.,
      'num_repeat': None,
      'guess': None,
      'bound': None,
      'model_func_name': 'monoexp',
      'model_func': None, #insert model function directly
      'save_name': '',
      'save_folder': 'save_corr', # folder to save output
      'weight': None,
      'sigma_av_corr': 0 ,
      'ind_break': None, #index for break criteria in known sigma algorithm
      'N_avrg': 1., #input signal is averaged N times
      'N_break': 100, #stop criterium number of iterations
      'sigdiff_break': 0.02, #stop criterium change in sigma
    }

    self.func_dict={
      'monoexp': self.diff_mono,
      'biexp': self.diff_biexp,
      'gamma' : self.diff_gamma,
      'kurtosis' : self.diff_kurtosis,
      'stretched' : self.diff_stretched,
      'test_lin': self.test_lin,
      #3D functions
      'monoexp_3D': self.monoexp_3D,
      'biexp_3D': self.biexp_3D,
      'gamma_3D': self.gamma_3D,
      'kurtosis_3D': self.kurtosis_3D,
      'stretched_3D' : self.stretched_3D,
    }

    self.guess_dict={
      #guess/starting values for different modelfunctions
      'monoexp': (1.,1.),
      'biexp': (1.,1.,.5,1.),
      'gamma' : (1.,1.,1.),
      'stretched' : (1.,1.,1.),
      'kurtosis' : (1.,1.,1.),
      'test_lin': (1.,1.),
      #3D functions
      'monoexp_3D': (1.,1.,1.,1.),
      'biexp_3D': (1.,1.,.5,1.,1.,.5,1.,1.,.5,1.),
      'gamma_3D' : (1.,1.,1.,1.,1.,1.,1.),
      'stretched_3D' : (1.,1.,1.,1.,1.,1.,1.),
      'kurtosis_3D' : (1.,1.,1.,1.,1.,1.,1.),
      }

    self.bound_dict={
      #guess/starting values for different modelfunctions
      'monoexp': (-np.inf,np.inf),
      'biexp': ([0.,0.,0.,0.],[10.,10.,1.,np.inf]),
      'gamma' : (-np.inf,np.inf),
      'stretched' : (-np.inf,np.inf),
      'kurtosis' : (-np.inf,np.inf),
      'test_lin': (-np.inf,np.inf),
      #3D functions
      'monoexp_3D': (-np.inf,np.inf),
      'biexp_3D': ([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                   [10.,10.,1.,10.,10.,1.,10.,10.,1.,np.inf]),
      'gamma_3D' : (-np.inf,np.inf),
      'stretched_3D' : (-np.inf,np.inf),
      'kurtosis_3D' : (-np.inf,np.inf),
      }

  def set_parameter(self,para_dict=None,set_sim=True):
    """
    para_dict: simulation parameters
    set_sim: set parameters signal generation
    """
    if para_dict is None:
      self.para_dict=self.default_para_dict
    else:
      self.para_dict=para_dict
      #added missing keys
      key_miss=[key for key in self.default_para_dict if key not in para_dict]
      for key in key_miss:
        self.para_dict[key]=self.default_para_dict[key]
      key_invalid=[key for key in self.para_dict if key not in
                   self.default_para_dict]
      if len(key_invalid)>0:
        print('Keys {} not valid'.format(key_invalid))
    if set_sim:
      self.sig=self.para_dict['sig'] #sigma of gaussian noise
      self.sim_para=self.para_dict['sim_para'] #input parameters
      self.SNR=self.para_dict['SNR']

    self.num_repeat=self.para_dict['num_repeat']
    self.b=self.para_dict['b']
    self.guess=self.para_dict['guess']
    self.bound=self.para_dict['bound']
    self.weight=self.para_dict['weight']
    self.sigma_av_corr=self.para_dict['sigma_av_corr']
    self.model_func_name=self.para_dict['model_func_name']
    self.model_func=self.para_dict['model_func']
    self.save_name=self.para_dict['save_name']
    self.save_folder=self.para_dict['save_folder']
    self.ind_break=self.para_dict['ind_break']
    self.N_avrg=self.para_dict['N_avrg']
    self.N_break=self.para_dict['N_break']
    self.sigdiff_break=self.para_dict['sigdiff_break']


  def gen_rice_signal(self,sig_input,sig,random_state=None):
    """
    para_dict: simulation parameters
    """
    shape_sig_input=sig_input.shape
    rice_sig=self.rice_rv(sig_input.flatten(),sig,np.prod(shape_sig_input),
                          random_state=random_state).reshape(shape_sig_input)
    return rice_sig


  def gen_gauss_signal(self,sig_input,sig,random_state=None):
    """
    generate model data first and then run bias algorithm
    sig_input: input signal
    sig: sigma of noise

    return: gauss noise inflicted signal
    """
    shape_sig_input=sig_input.shape
    gauss_sig=st.norm.rvs(sig_input.flatten(),sig,np.prod(shape_sig_input),
                          random_state=random_state).reshape(shape_sig_input)
    return gauss_sig

  def run_bias_corr_core(self,data,
                         guess,bound,
                         bias,sigma,
                         save_step,
                         para_arr,
                         **kwargs):
    """
    generate model data first and then run bias algorithm
    para_dict: simulation parameters
    data: run on preloaded signal numpy array
          if str then path to npy file with signal
    para_arr: array for saving results
    make_output: save output on disk
    sigma: run with known sigma, can either be float
           or numpy array (len must be data.shape[0])
    """
    #INITIALIZATION

    num_iter=data.shape[0]
    iter_run_arr=np.zeros(num_iter,dtype=int)
    #saving fitting errors (keep as list for the moment)
    para_err=[]

    if 'res_func' in kwargs:
      res_func=kwargs['res_func']
    else:
      res_func=None

    for j in np.arange(num_iter):
      if (j % (num_iter/10.) < 1. and self.show_progress and
      self.num_par==1):
        print('{:.0f}%'.format(np.round(j/num_iter*10)*10))
      rice_sig=data[j]
      guess_run=guess[j] if type(guess)==np.ndarray else guess
      bound_run=bound[j] if type(bound)==np.ndarray else bound

      #run with known sigma
      if type(sigma) in [list,np.ndarray]:
        sigma_run=sigma[j]
      else:
        sigma_run=sigma

      #run simulation
      if 'MLE' in kwargs and kwargs['MLE']:
        para_run=self.mle(self.model_func,self.b,
                          rice_sig,guess_run,bound_run,
                          sigma_run)
      else:
        if bias is not None:
          #run with known bias
          para_run=self.bias_corrector_direct(self.model_func,self.b,
                                              rice_sig,bias,sigma_run,
                                              guess_run,bound_run,self.weight,
          )
        elif sigma is not None:
          #run with known sigma
          para_run=self.bias_corrector_sigma(self.model_func,self.b,
                                             rice_sig,
                                             sigma_run,self.ind_break,
                                             guess_run,bound_run,self.weight,
                                             self.sigdiff_break,self.N_break,
          )
        else:
          #run full algorithm
          para_run=self.bias_corrector(self.model_func,self.b,
                                       rice_sig,guess_run,
                                       bound_run,res_func,
                                       self.weight,self.sigma_av_corr,
                                       self.N_avrg,
                                       self.sigdiff_break,
                                       self.N_break,
          )
      #add result to para_list
      if para_run is not None:
        num_it_run=para_run[2]
        iter_run_arr[j]=num_it_run
        if save_step:
          para_arr[j,:num_it_run+1]=para_run[0]
          para_err.append(para_run[1])
        else:
          para_arr[j,0]=para_run[0][num_it_run]
          para_err.append(para_run[1][num_it_run])

    if self.num_par>1:
      self.update_progress()
   
    return [para_arr,para_err,iter_run_arr]

  def run_bias_corr(self,para_sim,data,*args,
                    bias=None,sigma=None,
                    save_step=True,show_progress=True,
                    num_par=None,make_output=True,
                    **kwargs):
    """
    args and kwargs passed on to run_bias_corr_core
    num_cpu: number of parallel processes (if None: max num_cpu-1; min 1)
    """
    start=time.time()
    num_cpu_system=mp.cpu_count()
    if num_cpu_system==1:
      self.num_par=1
    if num_par is None:
      self.num_par=num_cpu_system-1
    else:
      self.num_par=num_par
    #set parameters for bias correction
    self.set_parameter(para_sim)

    #show progress
    self.show_progress=show_progress

    #create save folder
    if make_output and not os.path.exists(self.save_folder):
      os.makedirs(self.save_folder)

    #load signal
    if type(data)==np.ndarray:
      signal=data
      org_path_data=None
    elif type(data)==str:
      signal=np.load(data)
      org_path_data=data

    #INITIALIZATION
    #list with simulation results
    if self.num_repeat is None:
      num_iter=signal.shape[0]
    else:
      num_iter=int(np.min((self.num_repeat,signal.shape[0])))
      signal=signal[:num_iter]

    if not save_step or bias is not None or ('MLE' in kwargs and kwargs['MLE']):
      para_arr_len=1
    else:
      para_arr_len=self.N_break+1

    #set model_function
    if self.model_func is None:
      if self.model_func_name in self.func_dict:
        self.model_func=self.func_dict[self.model_func_name]
      else:
        print('Model function not found')
        return False
      if self.guess is None:
        self.guess=self.guess_dict[self.model_func_name]
      if self.bound is None:
        self.bound=self.bound_dict[self.model_func_name]
    else:
      if self.guess is None:
        print('Guess most be given for direct definition of model function')
        return False
      if self.bound is None:
        self.bound=(-np.inf,np.inf)

    if type(self.guess)==np.ndarray:
      assert(self.guess.shape[0]==signal.shape[0]),'Guess array wrong shape'
      num_res_par=self.guess.shape[-1]+1
    else:
      num_res_par=len(self.guess)+1

    if type(self.bound)==np.ndarray:
      assert(self.bound.shape[0]==signal.shape[0]),'Bound array wrong shape'

    if 'MLE' in kwargs and kwargs['MLE'] and sigma is None:
      #MLE includes sigma as parameter if sigma not given
      num_res_par-=1

    para_arr=np.zeros((num_iter,para_arr_len,num_res_par))
    #number of iterations
    iter_run_arr=np.zeros(num_iter,dtype=int)
    #saving fitting errors (keep as list for the moment)
    para_err=[]
      
    if self.num_par>1:
      """
      run parallel
      """
      #manager for parallel tasks
      manager=mp.Manager()
      self.num_chuck_processed=manager.Value('i',0)
      self.lock=manager.Lock()
      pool = mp.Pool(self.num_par)
      pool_list=[]
      num_profile=signal.shape[0]
      profile_per_chunk=20 # number of profiles per signal chunk
      min_task=10 # min number tasks chunks per cpu
      self.split_fac=(int(num_profile/profile_per_chunk)
                      if num_profile/profile_per_chunk>(min_task*
                                                        self.num_par)
                      else self.num_par)
      signal_split=np.array_split(signal,self.split_fac)
      if sigma is not None and type(sigma) not in [float,int,np.float64]:
        sigma_split=np.array_split(sigma,self.split_fac)
      else:
        sigma_split=None
      if type(self.guess)==np.ndarray:
        guess_split=np.array_split(self.guess,self.split_fac)
      else:
        guess_split=None
      if type(self.bound)==np.ndarray:
        bound_split=np.array_split(self.bound,self.split_fac)
      else:
        bound_split=None

      para_arr_split=np.array_split(para_arr,self.split_fac)
      iter_run_arr_split=np.array_split(iter_run_arr,self.split_fac)
      for i,signal_chunk in enumerate(signal_split):
        #show progress only for first cpu
        kw={**kwargs}
        #generate pools
        sigma_pool=sigma_split[i] if sigma_split is not None else sigma
        guess_pool=guess_split[i] if guess_split is not None else self.guess
        bound_pool=bound_split[i] if bound_split is not None else self.bound

        pool_list.append(pool.apply_async(self.run_bias_corr_core,
                                          (signal_chunk,
                                           guess_pool,bound_pool,
                                           bias,sigma_pool,
                                           save_step,
                                           para_arr_split[i])+args,
                                          kw))
      #collect results
      ind_sum=0
      for i_p,pool_res in enumerate(pool_list):
        try:
          para_res_core=pool_res.get()
        except KeyboardInterrupt:
          print("Interrupted by user")
          pool.terminate()
          break
        if para_res_core is not None:
          size_res=para_res_core[0].shape[0]
          para_arr[ind_sum:ind_sum+size_res]=para_res_core[0]
          iter_run_arr[ind_sum:ind_sum+size_res]=para_res_core[2]
          para_err.append(para_res_core[1])
          ind_sum+=size_res
        else:
          print('Problem with pools')
        para_res=[para_arr,para_err,iter_run_arr]
      pool.close() # shut down the pool

    else:
      """
      run one core
      """
      para_res=self.run_bias_corr_core(data,self.guess,self.bound,
                                       bias,sigma,save_step,
                                       para_arr,
                                       *args,**kwargs)
      iter_run_arr=para_res[2]

    #Check algorithmtype
    if 'bias' in kwargs and kwargs['bias'] is not None:
      algorithm_type='Bias'
    elif 'sigma' in kwargs and kwargs['sigma'] is not None:
      algorithm_type='Sigma'
    else:
      algorithm_type='Full'

    if save_step:
      max_iter=np.max(iter_run_arr)
      para_res[0]=para_res[0][:,:max_iter+1]
    if make_output:
      self.save_result(signal,para_res,
                       org_path_data,algorithm_type)
      
    if self.show_progress:
      duration_run=(time.time()-start) #duration of run in s
      minutes, seconds = divmod(duration_run, 60)
      time_str = '{:.0f} minutes {:.1f} seconds'.format(minutes, seconds)
      print('\nFinished after {}'.format(time_str))

    return para_res

  def update_progress(self,finished=True):
    if finished:
      with self.lock:
        self.num_chuck_processed.value+=1
    # number of chunks processed
    c_p=int(self.num_chuck_processed.value/self.split_fac*1e2)
    res_str='{:3d}%'.format(c_p)
    if self.show_progress and 1==1:
      print(res_str, end='\r')
  

  def save_result(self,signal,para_res,
                  org_path_data,alg_type):
    """
    """
    save_dict={}
    timestr=time.strftime("_%y%m%d_%H%M")
    save_string=self.save_name
    base_name_path=os.path.join(self.save_folder,save_string)
    info_string=''
    info_string+=('Correction run on {}\n'.format(timestr))
    if org_path_data is not None:
      info_string+=('Org data path {}\n'.format(org_path_data))
      save_dict['data']=org_path_data
    else:
      info_string+=('Org data path {}\n'.format(org_path_data))
      save_dict['data']=signal
    info_string+=('Algorithm type {}\n'.format(alg_type))
    info_string+=('Result file: {}.npy\n'.format(save_string))
    info_string+=('Signal file: {}_sig.npy\n'.format(save_string))
    save_dict['info_string']=info_string
    res_dict={
      'para_arr': para_res[0],
      'para_err': para_res[1],
      'nstep': para_res[2],
      }
    save_dict['res_dict']=res_dict
    save_dict['para_dict']=self.para_dict
    np.save(base_name_path+'.npy',save_dict)

  def bias_corrector(self,model_func,b,rice_sig,
                     guess,bound,
                     res_func=None,
                     weight=None,
                     num_sigmaes_avrg=0,
                     N_avrg=1,
                     sigdiff_break=.02,
                     N_break=100):
    """
    Algorithm Implementation
    Run Bias Correction
    model_func: underlying model function
    b: array of b-factors (b=gamma^2*g^2*delta^2*(Delta-delta/3)) [s/mm^2]
    rice_sig: signal to be corrected (Rician distributed)
    guess: initial guess for the fit funtion (must be set; not None)
    bound: bounds for the fit
    weight: use weighting of datapoints (rician)
    num_sigmaes_avrg: "number of freedom" for sigmaes calculation
    N_avrg: factor for actual sigma if signal is averaged
    """
    if res_func is None:
      def res_func(SNR): return self.alpha_func(SNR,N=N_avrg)

    rice_sig_corr=np.zeros_like(rice_sig)
    #set dimensions and number of fitting parameters
    num_dp=rice_sig.shape[0]
    num_fp=len(guess) #number of fitting parameters
    #effective number of datapoints for sigmaes calculation
    num_siges=num_dp-num_sigmaes_avrg
    #list for run parameters
    para_run_arr=np.zeros((self.N_break,len(guess)+1))
    #saving fitting errors (keep as list for the moment)
    para_run_err=[]

    #STEP 1
    #fit exp and obtain ŝ_0 (fit_sig) and RMSE
    res=so.curve_fit(model_func,b,rice_sig,guess,
                        bounds=bound,method=method_grad)
    if res is False:
      if self.debug:
          print('Problem with Fit!')
      return None
    fit_sig=model_func(b,*res[0]) # FIX parameter input
    RMSE=np.sqrt(np.sum((fit_sig-rice_sig)**2)/(num_dp-num_fp))
    #STEP 2
    #get RMSE
    sigma_corr_fac_RMSE=np.sqrt(N_avrg)
    sigma_es=RMSE*sigma_corr_fac_RMSE
    #save parameters of first fit and estimation
    para_run_arr[0,:-1]=res[0]
    para_run_arr[0,-1]=sigma_es
    para_run_err.append(res[1])
    for i in np.arange(1,N_break):
      #STEP 3 + 4
      #Estimate Rician bias and calculate corrected signal
      rice_sig_corr=rice_sig-(self.rice_mean_high(fit_sig,sigma_es)
                              -fit_sig)
      #STEP 5
      res=so.curve_fit(model_func,b,rice_sig_corr,guess,
                       bounds=bound,method=method_grad)
      if res is False:
        if self.debug:
          print('Problem with Fit!')
        return None
      fit_sig=model_func(b,*res[0])
      #step 6+7
      #get new alpha and estimate sigma
      SNR_es=fit_sig/sigma_es
      alpha=res_func(SNR_es)
      sigma_es_bf=np.copy(sigma_es)
      sigma_es=np.sum(np.abs(fit_sig-rice_sig)/alpha)/num_siges
      if 1==1:
        para_run_arr[i,:-1]=res[0]
        para_run_arr[i,-1]=sigma_es
        para_run_err.append(res[1])
      #step 8 stopping crit
      if np.abs(sigma_es-sigma_es_bf)/sigma_es<sigdiff_break:
        break
    return [para_run_arr[:i+1],para_run_err,i]

  def bias_corrector_sigma(self,model_func,b,rice_sig,
                           sigma,ind_break,
                           guess,bound,weight,
                           sigdiff_break=.02,
                           N_break=100):
    """
    Algorithm Implementation without sigma estimation
    Run Bias Correction
    model_func: underlying model function
    b: array of b-factors (b=gamma^2*g^2*delta^2*(Delta-delta/3)) [s/mm^2]
    rice_sig: signal to be corrected (Rician distributed)
    guess: initial guess for the fit funtion (must be set; not None)
    bound: bounds for the fit
    weight: use weighting of datapoints (rician)
    num_sigmaes_avrg: "number of freedom" for sigmaes calculation
    """
    #array for corrected signal
    rice_sig_corr=np.zeros_like(rice_sig)
    #set dimensions and number of fitting parameters
    num_dp=np.prod(b.shape) #number of datapoints in signal
    if ind_break is None:
      ind_break=int(num_dp/2.) # index for break criterium
    num_fp=len(guess) #number of fitting parameters
    #effective number of datapoints for sigmaes calculation
    #list for run parameters
    para_run_arr=np.zeros((self.N_break,len(guess)+1))
    #saving fitting errors (keep as list for the moment)
    para_run_err=[]


    #STEP 1
    #fit exp and obtain ŝ_0 (fit_sig) and RMSE
    res=so.curve_fit(model_func,b,rice_sig_corr,guess,
                     bounds=bound,method=method_grad,)
    if res is False:
      if self.debug:
          print('Problem with Fit!')
      return None
    fit_sig=model_func(b,*res[0]) # FIX parameter input
    #save parameters of first fit and estimation
    para_run_arr[0,:-1]=res[0]
    para_run_arr[0,-1]=sigma
    para_run_err.append(res[1])

    for i in np.arange(1,N_break):
      #STEP 3 + 4
      #Estimate Rician bias and calculate corrected signal
      rice_sig_corr=rice_sig-(self.rice_mean_high(fit_sig,sigma)
                              -fit_sig)
      #STEP 5
      #fit exp
      res=so.curve_fit(model_func,b,rice_sig_corr,guess,
                       bounds=bound,method=method_grad,)
      if res is False:
        if self.debug:
          print('Problem with Fit!')
        return None
      fit_sig_bf=np.copy(fit_sig)
      fit_sig=model_func(b,*res[0])
      #step 6
      if 1==1:
        para_run_arr[i,:-1]=res[0]
        para_run_arr[i,-1]=sigma
        para_run_err.append(res[1])
      #step 8 stopping crit
      if np.max(np.abs(fit_sig[ind_break]-
                fit_sig_bf[ind_break])/fit_sig[ind_break])<sigdiff_break:
        break
    return [para_run_arr[:i+1],para_run_err,i]

  def bias_corrector_direct(self,model_func,b,rice_sig,
                            bias,sigma,
                            guess,bound,weight):
    """
    Direct correction of bias via input array
    Run Bias Correction
    model_func: underlying model function
    b: array of b-factors (b=gamma^2*g^2*delta^2*(Delta-delta/3)) [s/mm^2]
    rice_sig: signal to be corrected (Rician distributed)
    bias: bias to correct for; either scalar or same dim as rice_sig
    sigma: sigma of distribution for weighting calculation
    guess: initial guess for the fit funtion (must be set; not None)
    bound: bounds for the fit
    weight: use weighting of datapoints (rician)
    num_sigmaes_avrg: "number of freedom" for sigmaes calculation
    """
    #corrected signal
    rice_sig_corr=rice_sig-bias
    #fit function into data
    res=so.curve_fit(model_func,b,rice_sig_corr,guess,
                     bounds=bound,method=method_grad,)
    #keep same output structure as for "bias_corrector"
    if res is False:
      if self.debug:
        print('Problem with Fit!')
      return None

    #calculate a sigma for fit
    num_dp=np.prod(b.shape) #number of datapoints in signal
    num_fp=len(guess) #number of fitting parameters
    fit_sig=model_func(b,*res[0])
    if num_dp>num_fp:
      RMSE=np.sqrt(np.sum((fit_sig-rice_sig_corr)**2)/(num_dp-num_fp))
    else:
      RMSE=0
    sigma_es=RMSE
    para_run=[[[*res[0],sigma_es]],res[1],0]
    return para_run

  def mle(self,model_func,b,rice_sig,
          guess,bound,sigma=None):
    """
    Maximum likelihood implementation
    Run Bias Correction
    model_func: underlying model function
    b: array of b-factors (b=gamma^2*g^2*delta^2*(Delta-delta/3)) [s/mm^2]
    rice_sig: signal to be corrected (Rician distributed)
    guess: initial guess for the fit funtion (must be set; not None)
    bound: bounds for the fit
    """
    #set dimensions and number of fitting parameters
    num_dp=np.prod(b.shape) #number of datapoints in signal
    num_fp=len(guess) #number of fitting parameters
    bound=so.Bounds(*bound)

    def func_min(p,x,y):
      """
      p: model parameters to be optimized + sigma (last parameter)
      x: b-values
      y: signal
      """
      if sigma is None:
        num_model=num_fp-1
        sigma_alg=p[-1]
      else:
        num_model=num_fp
        sigma_alg=sigma
      nu=model_func(x,*p[:num_model])
      return -np.sum(self.rice(y,nu,sigma_alg,log_out=True))

    res=so.minimize(func_min,guess,args=(b,rice_sig),bounds=bound,
                    method='L-BFGS-B',)
                    #method='Powell',)
    if res.success is False:
      if self.debug:
          print('Problem with Fit!')
      return None
    else:
      if sigma is None:
        result=res.x
      else:
        result=np.append(res.x,[sigma])
      return [[result],None,0]
