
import numpy as np
import ray
import scipy.stats as st

import sys
if sys.platform =='win32':
    import time as time
    import win_precise_time as ptime
elif sys.platform =='darwin':
    import time as time
    import time as ptime

# The TpMaxGenerator function in utils.py is slightly changed during the review process.
# The change does not impact the main conclusions.
# To produce exactly the same results, we create a utils_updated.py.


    
def parallel_experiments(rng,  generators, policy=None, remote_policy=None, delta=0.01, test_time=False, args={}):

    """
    Conducts parallel experiments using Ray.

    Args:
        rng (numpy.random.Generator): Random number generator.
        generators (list): List of generator instances.
        policy (function): Function for policy execution.
        remote_policy (function): Remote function for policy execution.
        delta (float): Tolerance for PGS evaluation.
        test_time (bool): Flag for testing time of a single replication.
        args (dict): Additional arguments.

    Returns:
        tuple: PCS and PGS values.
    """

    _start = time.time()
    results = []
    n_replications = len(generators)
    k = generators[0].syscount()
    print("--------New experiments with  k={}----------------------".format(k))
    for expe_id in range(n_replications):
        _seed = rng.integers(1, 10e8)
        if test_time:
            if expe_id == 0:
                start = time.time()
                policy(generators[expe_id],  **args)
                end = time.time()
                print("Single replication takes: {}s".format(end-start))
                print("Estimated serial total time: {}s".format((end-start)* n_replications))
        else:
            pass
        results.append(remote_policy.remote(generators[expe_id], seed=_seed, expe_id = expe_id, **args))
    print("Start to simulate... at {}".format(time.ctime()))
    results = ray.get(results)
    PCS, PGS = evaluate_PCS(generators, results), evaluate_PGS(generators, results, delta=delta)
    print("PCS:{}, PGS:{}".format(PCS, PGS))
    _end = time.time()
    print("Total time used: {}s, simulation ends at {}".format(_end-_start, time.ctime()))
    return PCS, PGS


class SCCVGenerator(object):
    def __init__(self, n_alternatives, gamma, var, best_index=0):
        """
        Initializes an instance of the SCCVGenerator class.

        Args:
            n_alternatives (int): Number of alternatives.
            gamma (float): Value assigned to the best alternative.
            var (float): Variance for all alternatives.
            best_index (int): Index of the best alternative (default is 0).
        """

        self.n_alternatives = n_alternatives
        self.gamma = gamma
        self.means = np.zeros(n_alternatives)
        self.means[best_index] = gamma
        self.best_mean = gamma
        self.variances = np.ones(self.n_alternatives)*var
        self.stds = np.sqrt(self.variances)
        
    def get(self, index, n=1):
        """
        Generates n samples from the normal distribution for the specified alternative index.

        Args:
            index (int): Index of the alternative.
            n (int): Number of samples to generate (default is 1).

        Returns:
            numpy.ndarray: Array of n samples.
        """
        return np.random.normal(self.means[index], self.stds[index], n)
        
    def syscount(self):
        """
        Returns the number of alternatives.

        Returns:
            int: Number of alternatives.
        """

        return self.n_alternatives
    
    
class EMCVGenerator(object):
    def __init__(self, n_alternatives, gamma, lamda, var, best_index=0):
        """
        Initializes an instance of the EMCVGenerator class.

        Args:
            n_alternatives (int): Number of alternatives.
            gamma (float): Value assigned to the best alternative.
            lamda (float): Parameter controlling the means of non-best alternatives.
            var (float): Variance for all alternatives.
            best_index (int): Index of the best alternative (default is 0).
        """

        self.n_alternatives = n_alternatives
        self.gamma = gamma
        self.means = np.arange(n_alternatives)/n_alternatives*lamda
        self.means[best_index] = gamma
        self.best_mean = gamma
        self.variances = np.ones(self.n_alternatives)*var
        self.stds = np.sqrt(self.variances)
        
    def get(self, index, n=1):
        return np.random.normal(self.means[index], self.stds[index], n)
        
    def syscount(self):
        return self.n_alternatives
    

class EMIVGenerator(object):
    def __init__(self, n_alternatives, gamma, lamda, varlow, varhigh, best_index=0):
        """
        Initializes an instance of the EMIVGenerator class.

        Args:
            n_alternatives (int): Number of alternatives.
            gamma (float): Value assigned to the best alternative.
            lamda (float): Parameter controlling the means of non-best alternatives.
            varlow (float): Lower bound of the variance for all alternatives.
            varhigh (float): Upper bound of the variance for all alternatives.
            best_index (int): Index of the best alternative (default is 0).
        """

        self.n_alternatives = n_alternatives
        self.gamma = gamma
        self.means = np.arange(n_alternatives)/n_alternatives*lamda
        self.means[best_index] = gamma
        self.best_mean = gamma
        
        self.variances = varlow + np.arange(n_alternatives)/n_alternatives*(varhigh-varlow)
        self.stds = np.sqrt(self.variances)
        
    def get(self, index, n=1):
        return np.random.normal(self.means[index], self.stds[index], n)
        
    def syscount(self):
        return self.n_alternatives
    

class EMDVGenerator(object):
    def __init__(self, n_alternatives, gamma, lamda, varlow, varhigh, best_index=0):
        """
        Initializes an instance of the EMDVGenerator class.

        Args:
            n_alternatives (int): Number of alternatives.
            gamma (float): Value assigned to the best alternative.
            lamda (float): Parameter controlling the means of non-best alternatives.
            varlow (float): Lower bound of the variance for all alternatives.
            varhigh (float): Upper bound of the variance for all alternatives.
            best_index (int): Index of the best alternative (default is 0).
        """

        self.n_alternatives = n_alternatives
        self.gamma = gamma
        self.means = np.arange(n_alternatives)/n_alternatives*lamda
        self.means[best_index] = gamma
        self.best_mean = gamma
        
        self.variances = varhigh + np.arange(n_alternatives)/n_alternatives*(varlow-varhigh)
        self.stds = np.sqrt(self.variances)
        
    def get(self, index, n=1):
        return np.random.normal(self.means[index], self.stds[index], n)
        
    def syscount(self):
        return self.n_alternatives

def maxG(k):
    """
    Computes the maximum number of groups (G) such that G*(2^G - 1) <= k.

    Args:
        k (int): Maximum value.

    Returns:
        int: Maximum number of groups (G).
    """

    _sum = 1
    G = 1
    while _sum < k:
        G += 1
        _sum  = G*(2**(G)-1)
    if _sum > k:
        G -= 1
    return G

def evaluate_PCS(generators, return_expe_best_ids):
    """
    Evaluate the Probability of Correct Selection (PCS) for a list of generators and best IDs.

    Args:
        generators (list): List of generator objects.
        return_expe_best_ids (list): List of tuples containing experiment IDs and estimated best IDs.

    Returns:
        float: Probability of Correct Selection (PCS).
    """

    size = len(return_expe_best_ids)
    n_correct = 0
    for i in range(size):
        expe_id = return_expe_best_ids[i][0]
        estimated_bestid = return_expe_best_ids[i][1]
        if generators[expe_id].means[estimated_bestid]  == generators[expe_id].best_mean:
            n_correct += 1
    return n_correct/size

    
def evaluate_PGS(generators, return_expe_best_ids, delta=0.01):
    """
    Evaluate the Probability of Good Selection (PGS) for a list of generators and best IDs.

    Args:
        generators (list): List of generator objects.
        return_expe_best_ids (list): List of tuples containing experiment IDs and estimated best IDs.
        delta (float): Tolerance parameter for good selection.

    Returns:
        float: Probability of Good Selection (PGS).
    """
    size = len(return_expe_best_ids)
    n_good = 0
    for i in range(size):
        expe_id = return_expe_best_ids[i][0]
        estimated_bestid = return_expe_best_ids[i][1]
        if generators[expe_id].best_mean - generators[expe_id].means[estimated_bestid]  < delta:
            n_good += 1
    return n_good/size 


def loadTPMax(L1, L2):
    """
    Load throughput maximization (TP) means from a CSV file for a particular problem instance.

    Args:
        L1 (int): Parameter for the number of stages (1 to L1-1).
        L2 (int): Parameter for the number of jobs (1 to L2-1).

    Returns:
        tuple: Tuple containing arrays of solution vectors (slns) and means.
    """

    total = int((L1 - 1) ** 2 * (L2-2)/2)
    means = np.loadtxt("results_{}_{}.csv".format(L1, L2), delimiter=",")
    count = 0
    slns = []
    for s1 in range(1, L1-1): # 1-18
        for s2 in range(1, L1-s1): # 1-18
            s3 = L1 - s1 - s2
            for s4 in range(1, L2): # 1-19
                s5 = L2 - s4
                sln = [s1, s2, s3, s4, s5]
                # true_values[sln] = results[count]
                slns.append(sln)
                count += 1
    slns = np.array(slns)
    return slns, means


def TpMax(_njobs,  _nstages, _burnin, r, b, simulator):
    """
    Generate throughput maximization (TP) simulations using the specified parameters.

    Args:
        _njobs (int): Number of jobs.
        _nstages (int): Number of stages.
        _burnin (int): Burn-in parameter.
        r (list): List of exponential rate parameters for each stage.
        b (list): List of b parameters for each stage.
        simulator (function): Simulator function for TP.

    Returns:
        ndarray: Array of TP simulation results.
    """

    sTime1 = np.random.exponential(1/r[0], _njobs)
    sTime2 = np.random.exponential(1/r[1], _njobs)
    sTime3 = np.random.exponential(1/r[2], _njobs)
    b1 = b[0]
    b2 = b[1]
    return simulator(_njobs, _burnin, sTime1, sTime2, sTime3, b1, b2)

    
# Rewrite the TpMaxGenerator function of the utils.py
# The __init__ is slightly updated compared to the original version during the review process.
# To produce exactly the same results, we write this version here.

class TpMaxGenerator(object):
    """
    Generator class for throughput maximization (TP) simulations.

    Attributes:
        _njobs (int): Number of jobs.
        _nstages (int): Number of stages.
        _burnin (int): Burn-in parameter.
        slns (ndarray): Array of solution vectors.
        means (ndarray): Array of means.
        best_mean (float): Maximum mean value.
        simulator (function): Simulator function for TP.
    """

    def __init__(self, _njobs,  _nstages, _burnin, slns, means, simulator):
        """
        Initialize the TpMaxGenerator with the provided parameters.

        Args:
            _njobs (int): Number of jobs.
            _nstages (int): Number of stages.
            _burnin (int): Burn-in parameter.
            slns (ndarray): Array of solution vectors.
            means (ndarray): Array of means.
            simulator (function): Simulator function for TP.
        """

        self._njobs = _njobs
        self._nstages = _nstages
        self._burnin = _burnin
        self.slns = slns
        self.means = means
        self.best_mean = np.max(self.means)
        self.simulator = simulator
        
    def get(self, index, n=1):
        """
        Generate TP simulations based on the specified index.

        Args:
            index (int): Index of the solution vector.
            n (int): Number of simulations to generate (default is 1).

        Returns:
            ndarray: Array of TP simulation results.
        """

        r = self.slns[index][:self._nstages]
        b = self.slns[index][self._nstages:]
        if n == 1:
            return TpMax(self._njobs, self._nstages, self._burnin, r, b, self.simulator)
        else:
            results = [TpMax(self._njobs, self._nstages, self._burnin, r, b, self.simulator) for i in range(n)]
            return np.array(results)
        
    def syscount(self):
        return len(self.slns)
    
    def getbest(self):
        return self.best_mean


def TpMax_pause(_njobs, _nstages, _burnin, r, b, simulator):
    """
    Generate TP simulations with a pause.

    Args:
        _njobs (int): Number of jobs.
        _nstages (int): Number of stages.
        _burnin (int): Burn-in parameter.
        r (list): List of exponential rates.
        b (list): List of burn-in parameters.
        simulator (function): Simulator function for TP.

    Returns:
        ndarray: Array of TP simulation results.
    """
    # Pause for a random duration
    ptime.sleep(0.0005 + np.random.rand() * 0.0015)
    
    # Generate random exponential times
    sTime1 = np.random.exponential(1 / r[0], _njobs)
    sTime2 = np.random.exponential(1 / r[1], _njobs)
    sTime3 = np.random.exponential(1 / r[2], _njobs)
    
    b1 = b[0]
    b2 = b[1]
    
    # Perform TP simulation using the provided simulator
    return simulator(_njobs, _burnin, sTime1, sTime2, sTime3, b1, b2)

    
class TpMaxGeneratorPause(object):
    def __init__(self, _njobs,  _nstages, _burnin, slns, means, simulator):
        self._njobs = _njobs
        self._nstages = _nstages
        self._burnin = _burnin
        ids = np.arange(len(slns))
        self.slns = slns
        self.means = means
        self.best_mean = np.max(self.means)
        self.simulator = simulator
        
    def get(self, index, n=1):
        r = self.slns[index][:self._nstages]
        b = self.slns[index][self._nstages:]
        if n == 1:
            return TpMax_pause(self._njobs, self._nstages, self._burnin, r, b, self.simulator)
        else:
            results = [TpMax_pause(self._njobs, self._nstages, self._burnin, r, b, self.simulator) for i in range(n)]
            return np.array(results)
        
    def syscount(self):
        return len(self.slns)
    
    def getbest(self):
        return self.best_mean

def evaluate_PCS_parallel(generator, results):
        size = len(results)
        n_correct = 0
        for i in range(size):
            if generator.means[results[i]]  == generator.best_mean:
                n_correct += 1
        return n_correct/size
    

def evaluate_PGS_parallel(generator, results, delta=0.01):
        size = len(results)
        n_good = 0
        for i in range(size):
            if generator.best_mean - generator.means[results[i]]  < delta:
                            n_good += 1 
        return n_good/size
    

def process_result(generator, num_cores, results, phasetimeset, simutimeset):
    """
    Process the results of parallel experiments and print relevant information.

    Args:
        generator: Generator object for the experiment.
        num_cores (int): Number of computing threads used for parallel experiments.
        results (list): List of results from parallel experiments.
        phasetimeset (list): List of lists containing phase timing information.
        simutimeset (list): List of lists containing simulation timing information.
    """
    # Extract phase timing information
    seeding_times, exploration_times, greedy_times = phasetimeset
    simu_times1, simu_times2, simu_times3 = simutimeset

    # Calculate mean times for each phase
    seeding = np.mean(seeding_times)
    exploration = np.mean(exploration_times)
    greedy = np.mean(greedy_times)
    simu1 = np.mean(simu_times1)
    simu2 = np.mean(simu_times2)
    simu3 = np.mean(simu_times3)

    # Calculate utilization ratios for each phase
    util1 = simu1 / seeding / num_cores
    util2 = simu2 / exploration / num_cores
    util3 = simu3 / greedy / num_cores
    print("##### Utilization ratios for the stages: {:.2%}, {:.2%}, {:.2%} #####".format(util1, util2, util3))

    # Calculate overall procedure utilization
    util = (simu1 + simu2 + simu3) / (seeding + exploration + greedy) / num_cores
    print("##### Procedure utilization: {:.2%} #####".format(util))

    # Evaluate PCS and PGS
    PCS = evaluate_PCS_parallel(generator, results)
    PGS = evaluate_PGS_parallel(generator, results)
    print("******* PCS: {}, PGS: {} *******".format(PCS, PGS))

    # Print detailed timing information
    names = ["seeding time:", "exploration time:", "greedy time:", "simulation time in seeding:",
             "simulation time in exploration:", "simulation time in greedy:",
             "total time:", "total simulation time:"]
    timeset = phasetimeset + simutimeset
    timeset.append(np.sum(phasetimeset, axis=0))
    timeset.append(np.sum(simutimeset, axis=0))
    for i, times in enumerate(timeset):
        average = np.mean(times)
        CI = st.norm.interval(alpha=0.95, loc=np.mean(times), scale=st.sem(times))
        print(names[i] + " {:.3f} ".format(average) + u"\u00B1" + " {:.3f}".format(average - CI[0]))