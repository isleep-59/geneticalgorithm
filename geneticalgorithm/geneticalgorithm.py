'''

Copyright 2020 Ryan (Mohammad) Solgi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

'''

###############################################################################
###############################################################################
###############################################################################

import numpy as np
import sys
import time
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt
import os

###############################################################################
###############################################################################
###############################################################################


def evaluate_individuals(args):
    """Helper function for parallel evaluation."""
    individuals, func, timeout, f_kwargs = args
    results = func_timeout(timeout, func, args=(individuals, f_kwargs)) 
    return results

def evaluate_individual(args):
    """Non Concurrent version of evaluate_individuals."""
    individual, func, timeout, f_kwargs = args
    result = func_timeout(timeout, func, args=(individual, f_kwargs)) 
    return result

class geneticalgorithm():
    
    '''  Genetic Algorithm (Elitist version) for Python
    
    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.
    
    
    
    Implementation and output:
        
        methods:
                run(): implements the genetic algorithm
                
        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }
            
                report: a list including the record of the progress of the
                algorithm over iterations

    '''
    #############################################################
    def __init__(self, function, dimension, variable_type='bool', \
                 variable_boundaries=None,\
                 variable_type_mixed=None, \
                 function_timeout=10,\
                 algorithm_parameters={'max_num_iteration': None,\
                                       'population_size':100,\
                                       'mutation_probability':0.1,\
                                       'elit_ratio': 0.01,\
                                       'crossover_probability': 0.5,\
                                       'parents_portion': 0.3,\
                                       'crossover_type':'uniform',\
                                       'mutation_type' : 'single',\
                                       'init_type':'random',\
                                       'max_iteration_without_improv':None,\
                                        'improvement_threshold': 0.01,\
                                        'concurrent_processes': False},\
                     convergence_curve=True,\
                         progress_bar=True,\
                         log_limit=10,\
                         log_directory=None,\
                            **kwargs):


        '''
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function. 
        (For maximization multiply function by a negative sign: the absolute 
        value of the output would be the actual objective function)
        
        @param dimension <integer> - the number of decision variables
        
        @param variable_type <string> - 'bool' if all variables are Boolean; 
        'int' if all variables are integer; and 'real' if all variables are
        real value or continuous (for mixed type see @param variable_type_mixed)
        
        @param variable_boundaries <numpy array/None> - Default None; leave it 
        None if variable_type is 'bool'; otherwise provide an array of tuples 
        of length two as boundaries for each variable; 
        the length of the array must be equal dimension. For example, 
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first 
        and upper boundary 200 for second variable where dimension is 2.
        
        @param variable_type_mixed <numpy array/None> - Default None; leave it 
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first 
        variable is integer but the second one is real the input is: 
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1] 
        in variable_boundaries. Also if variable_type_mixed is applied, 
        variable_boundaries has to be defined.
        
        @param function_timeout <float> - if the given function does not provide 
        output before function_timeout (unit is seconds) the algorithm raise error.
        For example, when there is an infinite loop in the given function. 
        
        @param algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int> 
            @ mutation_probability <float in [0,1]>
            @ elit_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string> - Default is 'uniform'; 'one_point';
            'two_point' or 'pmx' are other options
            @ mutation_type <string> - Default is 'single'; 'inversion' or 'swap' are other options
            @ init_type <string> - Default is 'random'; 'sequential' is the other option
            @ max_iteration_without_improv <int> - maximum number of 
            successive iterations without improvement. If None it is ineffective
            @ improvement_threshold <float> - the threshold of improvement. If None it is ineffective
            @ concurrent_processes <True/False> - whether to use concurrent processes for function evaluation. Default is False.
        
        @param convergence_curve <True/False> - Plot the convergence curve or not
        Default is True.
        @progress_bar <True/False> - Show progress bar or not. Default is True.
        
        for more details and examples of implementation please visit:
            https://github.com/rmsolgi/geneticalgorithm
  
        '''
        self.__name__=geneticalgorithm
        #############################################################
        # input function
        assert (callable(function)),"function must be callable"     
        
        self.f=function
        #############################################################
        #dimension
        
        self.dim=int(dimension)
        
        #############################################################
        # input variable type
        
        assert(variable_type=='bool' or variable_type=='int' or\
               variable_type=='real'), \
               "\n variable_type must be 'bool', 'int', or 'real'"
       #############################################################
        # input variables' type (MIXED)     

        if variable_type_mixed is None:
            
            if variable_type=='real': 
                self.var_type=np.array([['real']]*self.dim)
            else:
                self.var_type=np.array([['int']]*self.dim)            

 
        else:
            assert (type(variable_type_mixed).__module__=='numpy'),\
            "\n variable_type must be numpy array"  
            assert (len(variable_type_mixed) == self.dim), \
            "\n variable_type must have a length equal dimension."       

            for i in variable_type_mixed:
                assert (i=='real' or i=='int'),\
                "\n variable_type_mixed is either 'int' or 'real' "+\
                "ex:['int','real','real']"+\
                "\n for 'boolean' use 'int' and specify boundary as [0,1]"
                

            self.var_type=variable_type_mixed
        #############################################################
        # input variables' boundaries 

            
        if variable_type!='bool' or type(variable_type_mixed).__module__=='numpy':
                       
            assert (type(variable_boundaries).__module__=='numpy'),\
            "\n variable_boundaries must be numpy array"
        
            assert (len(variable_boundaries)==self.dim),\
            "\n variable_boundaries must have a length equal dimension"        
        
        
            for i in variable_boundaries:
                assert (len(i) == 2), \
                "\n boundary for each variable must be a tuple of length two." 
                assert(i[0]<=i[1]),\
                "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound=variable_boundaries
        else:
            self.var_bound=np.array([[0,1]]*self.dim)
 
        ############################################################# 
        #Timeout
        self.funtimeout=float(function_timeout)
        ############################################################# 
        #convergence_curve
        if convergence_curve==True:
            self.convergence_curve=True
        else:
            self.convergence_curve=False
        ############################################################# 
        #progress_bar
        if progress_bar==True:
            self.progress_bar=True
        else:
            self.progress_bar=False
        ############################################################# 
        ############################################################# 
        # input algorithm's parameters
        
        self.param=algorithm_parameters
        
        self.pop_s=int(self.param['population_size'])
        
        assert (self.param['parents_portion']<=1\
                and self.param['parents_portion']>=0),\
        "parents_portion must be in range [0,1]" 
        
        self.par_s=int(self.param['parents_portion']*self.pop_s)
        trl=self.pop_s-self.par_s
        if trl % 2 != 0:
            self.par_s+=1
               
        self.prob_mut=self.param['mutation_probability']
        
        assert (self.prob_mut<=1 and self.prob_mut>=0), \
        "mutation_probability must be in range [0,1]"
        
        
        self.prob_cross=self.param['crossover_probability']
        assert (self.prob_cross<=1 and self.prob_cross>=0), \
        "mutation_probability must be in range [0,1]"
        
        assert (self.param['elit_ratio']<=1 and self.param['elit_ratio']>=0),\
        "elit_ratio must be in range [0,1]"                
        
        trl=self.pop_s*self.param['elit_ratio']
        if trl<1 and self.param['elit_ratio']>0:
            self.num_elit=1
        else:
            self.num_elit=int(trl)
            
        assert(self.par_s>=self.num_elit), \
        "\n number of parents must be greater than number of elits"
        
        if self.param['max_num_iteration']==None:
            self.iterate=0
            for i in range (0,self.dim):
                if self.var_type[i]=='int':
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*self.dim*(100/self.pop_s)
                else:
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*50*(100/self.pop_s)
            self.iterate=int(self.iterate)
            if (self.iterate*self.pop_s)>10000000:
                self.iterate=10000000/self.pop_s
        else:
            self.iterate=int(self.param['max_num_iteration'])
        
        self.c_type=self.param['crossover_type']
        assert (self.c_type=='uniform' or self.c_type=='one_point' or\
                self.c_type=='two_point' or self.c_type=='pmx'),\
        "\n crossover_type must 'uniform', 'one_point', 'two_point', or 'pmx' Enter string" 

        self.m_type=self.param['mutation_type']
        assert (self.m_type=='single' or self.m_type=='inversion' or self.m_type=='swap'),\
        "\n mutation_type must 'single', 'inversion', or 'swap' Enter string"

        self.init_type=self.param['init_type']
        assert (self.init_type=='random' or self.init_type=='sequential'),\
        "\n init_type must 'random' or 'sequential' Enter string"
        
        self.stop_mniwi=False
        if self.param['max_iteration_without_improv']==None:
            self.mniwi=self.iterate+1
        else: 
            self.mniwi=int(self.param['max_iteration_without_improv'])

        self.threshold=self.param['improvement_threshold']
        self.is_concurrent_processes = algorithm_parameters.get('concurrent_processes')
        
        self.kwargs=kwargs
        self.log_limit = log_limit
        self.log_directory = log_directory

        #############################################################

    def check_duplicates(self, x):
        return len(x) == len(set(x))

        
        ############################################################# 
    def run(self):
        
        
        ############################################################# 
        # Initial Population
        
        self.integers=np.where(self.var_type=='int')
        self.reals=np.where(self.var_type=='real')
        
        
        
        pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        solo=np.zeros(self.dim+1)
        var=np.zeros(self.dim)
        initial_population_vars = []

        for p in range(0,self.pop_s):
            # if self.init_type == 'random':
            #     var = np.random.permutation(196)
            # elif self.init_type == 'sequential':
            #     var = np.arange(196)
            if p == 0 and self.init_type == 'sequential':
                var = np.arange(196)
            else:
                var = np.random.permutation(196)
            solo[:self.dim] = var.copy()
            # assert(self.check_duplicates(var)), "gene shouldn't be duplicated." 
            initial_population_vars.append(var.copy())
         
            # for i in self.integers[0]:
            #     var[i]=np.random.randint(self.var_bound[i][0],\
            #             self.var_bound[i][1]+1)  
            #     solo[i]=var[i].copy()
            # for i in self.reals[0]:
            #     var[i]=self.var_bound[i][0]+np.random.random()*\
            #     (self.var_bound[i][1]-self.var_bound[i][0])    
            #     solo[i]=var[i].copy()


            # obj=self.sim(var)            
            # solo[self.dim]=obj
            # pop[p]=solo.copy()

        # tasks = [(var, self.f, self.funtimeout, self.kwargs) for var in initial_population_vars]
        # if self.is_concurrent_processes:    
        #     with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor: 
        #         results = list(executor.map(evaluate_individual, tasks))
        # else:
        #     results = [evaluate_individual(task) for task in tasks]
        if self.is_concurrent_processes:
            tasks = (initial_population_vars, self.f, self.funtimeout, self.kwargs)
            results = evaluate_individuals(tasks).cpu().numpy().tolist()
        else:
            tasks = [(var, self.f, self.funtimeout, self.kwargs) for var in initial_population_vars]
            results = [evaluate_individual(task) for task in tasks]

        for p in range(self.pop_s):
            pop[p, :self.dim] = initial_population_vars[p]
            pop[p, self.dim] = results[p]
        #############################################################

        #############################################################
        # Report
        self.report=[]
        self.log=[]
        pop = pop[pop[:,self.dim].argsort()]
        self.best_variable = pop[0,: self.dim].copy()
        self.best_function = pop[0,self.dim]
        ##############################################################   
                        
        t=1
        counter=0
        while t<=self.iterate:
            
            if self.progress_bar==True:
                self.progress(t,self.iterate,status="GA is running...")
            #############################################################
            #Sort
            pop = pop[pop[:,self.dim].argsort()]
                          
            if pop[0,self.dim]<self.best_function:
                counter=0
                self.best_function=pop[0,self.dim].copy()
                self.best_variable=pop[0,: self.dim].copy()
            else:
                counter+=1
            #############################################################
            # Report

            self.report.append(pop[0,self.dim])
            self.log.append(pop)

            re=np.array(self.report)
            self.write_log(re)
            ##############################################################         
            # Normalizing objective function 
            
            normobj=np.zeros(self.pop_s)
            
            minobj=pop[0,self.dim]
            if minobj<0:
                normobj=pop[:,self.dim]+abs(minobj)
                
            else:
                normobj=pop[:,self.dim].copy()
    
            maxnorm=np.amax(normobj)
            normobj=maxnorm-normobj+1

            #############################################################        
            # Calculate probability
            
            sum_normobj=np.sum(normobj)
            prob=np.zeros(self.pop_s)
            prob=normobj/sum_normobj
            cumprob=np.cumsum(prob)
  
            #############################################################        
            # Select parents
            par=np.array([np.zeros(self.dim+1)]*self.par_s)
            
            for k in range(0,self.num_elit):
                par[k]=pop[k].copy()
            for k in range(self.num_elit,self.par_s):
                index=np.searchsorted(cumprob,np.random.random())
                par[k]=pop[index].copy()
                
            ef_par_list=np.array([False]*self.par_s)
            par_count=0
            while par_count==0:
                for k in range(0,self.par_s):
                    if np.random.random()<=self.prob_cross:
                        ef_par_list[k]=True
                        par_count+=1
                 
            ef_par=par[ef_par_list].copy()
    
            #############################################################  
            #New generation
            offspring_vars = []
            num_offspring_to_generate = self.pop_s - self.par_s
            current_offspring_count = 0
            while current_offspring_count < num_offspring_to_generate:
                r1=np.random.randint(0,par_count)
                r2=np.random.randint(0,par_count)
                pvar1=ef_par[r1,: self.dim].copy()
                pvar2=ef_par[r2,: self.dim].copy()
                
                ch=self.cross(pvar1,pvar2,self.c_type)
                ch1=ch[0].copy()
                ch2=ch[1].copy()
                
                ch1=self.mut(ch1, self.m_type)   
                ch2=self.mut(ch2, self.m_type)

                offspring_vars.append(ch1)
                current_offspring_count += 1
                if current_offspring_count < num_offspring_to_generate:
                    offspring_vars.append(ch2)
                    current_offspring_count += 1

            # tasks = [(var, self.f, self.funtimeout, self.kwargs) for var in offspring_vars]
            # if self.is_concurrent_processes:
            #     with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            #         results = list(executor.map(evaluate_individual, tasks))
            # else:
            #     results = [evaluate_individual(task) for task in tasks]
            if self.is_concurrent_processes:
                tasks = (offspring_vars, self.f, self.funtimeout, self.kwargs)
                results = evaluate_individuals(tasks).cpu().numpy().tolist()
            else:
                tasks = [(var, self.f, self.funtimeout, self.kwargs) for var in offspring_vars]
                results = [evaluate_individual(task) for task in tasks]

            new_pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            for k in range(self.par_s):
                new_pop[k] = par[k].copy()

            for k in range(len(offspring_vars)):
                new_pop[self.par_s + k, :self.dim] = offspring_vars[k]
                new_pop[self.par_s + k, self.dim] = results[k]
                 
            pop = new_pop

            # pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            
            # for k in range(0,self.par_s):
            #     pop[k]=par[k].copy()
                
            # for k in range(self.par_s, self.pop_s, 2):
            #     r1=np.random.randint(0,par_count)
            #     r2=np.random.randint(0,par_count)
            #     pvar1=ef_par[r1,: self.dim].copy()
            #     pvar2=ef_par[r2,: self.dim].copy()
                
            #     ch=self.cross(pvar1,pvar2,self.c_type)
            #     ch1=ch[0].copy()
            #     ch2=ch[1].copy()
                
            #     ch1=self.mut(ch1, self.m_type)
            #     # ch2=self.mutmidle(ch2,pvar1,pvar2)     
            #     ch2=self.mut(ch2, self.m_type)          
            #     solo[: self.dim]=ch1.copy()                
            #     obj=self.sim(ch1)
            #     solo[self.dim]=obj
            #     pop[k]=solo.copy()                
            #     solo[: self.dim]=ch2.copy()                
            #     obj=self.sim(ch2)               
            #     solo[self.dim]=obj
            #     pop[k+1]=solo.copy()
        #############################################################       
            t+=1
            if counter > self.mniwi:
                pop = pop[pop[:,self.dim].argsort()]
                if pop[0,self.dim]>=self.best_function:
                    t=self.iterate
                    if self.progress_bar==True:
                        self.progress(t,self.iterate,status="GA is running...")
                    time.sleep(2)
                    t+=1
                    self.stop_mniwi=True

            # if no significant improvement in the last mniwi iterations
            # if np.size(self.report) >= self.mniwi and pop[0,self.dim] > self.report[-(self.mniwi-1)] - self.threshold:
            #     pop = pop[pop[:,self.dim].argsort()]
            #     if pop[0,self.dim]>=self.best_function:
            #         t=self.iterate
            #         if self.progress_bar==True:
            #             self.progress(t,self.iterate,status="GA is running...")
            #         time.sleep(2)
            #         t+=1
            #         self.stop_mniwi=True
                
                
        #############################################################
        #Sort
        pop = pop[pop[:,self.dim].argsort()]
        
        if pop[0,self.dim]<self.best_function:
                
            self.best_function=pop[0,self.dim].copy()
            self.best_variable=pop[0,: self.dim].copy()
        #############################################################
        # Report

        self.report.append(pop[0,self.dim])
        self.log.append(pop)
        valid_solutions = self.get_valid_solutions(self.log, threshold=0.7, num_inter=10)
        
 
        self.output_dict={'variable': self.best_variable,\
                          'function':self.best_function,\
                          'fitness_per_inter':self.report,\
                          'valid_solutions':valid_solutions}
                                
        if self.progress_bar==True:
            show=' '*100
            sys.stdout.write('\r%s' % (show))
        # sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        # sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        sys.stdout.flush() 
        re=np.array(self.report)
        self.write_log(re)
        
        if self.stop_mniwi==True:
            sys.stdout.write('\nWarning: GA is terminated due to the'+\
                             ' maximum number of iterations without improvement was met!')
##############################################################################         
##############################################################################         
    def cross(self,x,y,c_type):
         
        ofs1=x.copy()
        ofs2=y.copy()

        if c_type=='pmx':

            start, end = sorted(np.random.choice(self.dim, 2, replace=False))
            ofs1[start:end+1] = y[start:end+1].copy()
            ofs2[start:end+1] = x[start:end+1].copy()
            for i in range(0, self.dim):
                if i < start or i > end:
                    while ofs1[i] in ofs1[start:end+1]:
                        tmp = np.where(ofs1[start:end+1] == ofs1[i])[0][0] + start
                        ofs1[i] = ofs2[tmp]

                    while ofs2[i] in ofs2[start:end+1]:
                        tmp = np.where(ofs2[start:end+1] == ofs2[i])[0][0] + start
                        ofs2[i] = ofs1[tmp]

        elif c_type=='one_point':

            ran=np.random.randint(0,self.dim)
            for i in range(0,ran):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()

        elif c_type=='two_point':
                
            ran1=np.random.randint(0,self.dim)
            ran2=np.random.randint(ran1,self.dim)
                
            for i in range(ran1,ran2):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
        elif c_type=='uniform':
                
            for i in range(0, self.dim):
                ran=np.random.random()
                if ran <0.5:
                    ofs1[i]=y[i].copy()
                    ofs2[i]=x[i].copy() 
        
        assert(self.check_duplicates(ofs1)), "gene shouldn't be duplicated." 
        assert(self.check_duplicates(ofs2)), "gene shouldn't be duplicated." 
                   
        return np.array([ofs1,ofs2])
###############################################################################  
    
    def mut(self,x,m_type):
        ran=np.random.random()
        if ran < self.prob_mut:
            if m_type=='inversion':
                
                start, end = sorted(np.random.choice(self.dim, 2, replace=False))
                x[start:end+1] = x[start:end+1][::-1]

            elif m_type=='swap':

                idx1, idx2 = sorted(np.random.choice(self.dim, 2, replace=False))
                x[idx1], x[idx2] = x[idx2], x[idx1]

            elif m_type=='single':

                for i in self.integers[0]:
                    ran=np.random.random()
                    if ran < self.prob_mut:
                        x[i]=np.random.randint(self.var_bound[i][0],self.var_bound[i][1]+1) 

                for i in self.reals[0]:                
                    ran=np.random.random()
                    if ran < self.prob_mut:   
                        x[i]=self.var_bound[i][0]+np.random.random()*(self.var_bound[i][1]-self.var_bound[i][0])  

        assert(self.check_duplicates(x)), "gene shouldn't be duplicated."                  
            
        return x
###############################################################################
    def mutmidle(self, x, p1, p2):
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                if p1[i]<p2[i]:
                    x[i]=np.random.randint(p1[i],p2[i])
                elif p1[i]>p2[i]:
                    x[i]=np.random.randint(p2[i],p1[i])
                else:
                    x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1)
                        
        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   
                if p1[i]<p2[i]:
                    x[i]=p1[i]+np.random.random()*(p2[i]-p1[i])  
                elif p1[i]>p2[i]:
                    x[i]=p2[i]+np.random.random()*(p1[i]-p2[i])
                else:
                    x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0]) 
        return x
###############################################################################     
    def evaluate(self):
        return self.f(self.temp, self.kwargs)
###############################################################################    
    def sim(self,X):
        self.temp=X.copy()
        obj=None
        try:
            obj=func_timeout(self.funtimeout,self.evaluate)
        except FunctionTimedOut:
            print("given function is not applicable")
        assert (obj!=None), "After "+str(self.funtimeout)+" seconds delay "+\
                "func_timeout: the given function does not provide any output"
        return obj

###############################################################################
    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()     
###############################################################################
    def get_valid_solutions(self,results, threshold=0.7,num_inter=10):
        """
        获取有效的解决方案
        - results: 结果列表
        - threshold: 阈值
        - num_inter: 选取代数
        """
        valid_solutions = []
        num=min(len(results), num_inter)
        for i in range(1, num+1):
            pop=results[-i]
            for p in pop:
                if p[self.dim] < threshold:
                    valid_solutions.append(p[0:self.dim])
        return valid_solutions
###############################################################################       
    def write_log(self, re):
        # dir_name = f'{self.init_type}_{self.current_time}'
        # dir_path = f'/workspace/Order-Matters/Vision-RWKV/classification/ga4order_logs/{dir_name}'
        dir_path = f'{self.log_directory}'
        os.makedirs(dir_path, exist_ok=True)
        
        if self.convergence_curve==True:
            plt.plot(re)
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            # plt.show()
            plt.savefig(f'{dir_path}/convergence_curve.png')
            plt.clf()

        # np.savetxt(f'{dir_path}/best_solution.npy', self.best_variable)
        np.save(f'{dir_path}/best_solution.npy', self.best_variable)
        np.save(f'{dir_path}/valid_solutin.npy',np.unique(np.array(self.get_valid_solutions(self.log, threshold=0.7, num_inter=10)),axis=0))
        with open(f'{dir_path}/log.txt', 'w') as f:
            f.write('The best solution found:\n %s\n\n Objective function:\n %s\n\n' % (self.best_variable, self.best_function))
            idx = 0
            for item in self.log:
                f.write(f'Generation {idx}: {item[0, self.dim]}\n')

                limit = 0
                for i in item:
                    if(limit < self.log_limit):
                        # f.write("%s\n" % i)
                        f.write(f'\tsolution {limit+1}: {i[self.dim]}\n')
                        f.write(f'\t')
                        np.savetxt(f, i[:self.dim].reshape(1, -1), fmt='%d', delimiter=',')
                        f.write('\n')
                    limit += 1
                idx += 1
                f.write('\n')
            f.write('\n')
###############################################################################
