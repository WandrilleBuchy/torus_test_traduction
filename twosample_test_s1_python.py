#' Two-sample goodness-of-fit on the circle
#'
#' Performs a two-sample goodness-of-fit test for measures supported on the circle, based on a
#' distribution-free Wasserstein statistic.
#'
#' @param x numerical vector of sample in [0,1).
#' @param y numerical vector of sample in [0,1).
#' @param sim_null The simulated null distribution of the statistic. If NULL, the distribution is simulated with the given parameters (very time consuming).
#' @param NR The number of replicas if simulation is required.
#' @param NC The number of cores if parallel simulation is required.
#' @param n The sample sizes of the simulated samples is simulation is required.
#' 
#' @return
#' \itemize{
#'   \item stat - The test statistic.
#'   \item pvalue - The test p-value.
#' }
#'
#' @references [1] Delon, J., Salomon, J., Sobolevskii, A.: Fast transport optimization for Monge costs on the circle. SIAM J. Appl. Math. 70(7), 2239–2258 (2010).
#' [2] RAMDAS, A., GARCIA, N. and CUTURI, M. (2015). On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests. Entropy 19.
#' 
#' @examples
#' 
#' n = 50 # Sample size
#' 
#' # Simulate the statistic null distribution
#'
#' from scipy.stats import vonmises
#' import numpy as np
#'
#' NR = 100
#' sim_free_null = sim.null.stat(500, NC = 2)
#' 
#' x, y = np.random.rand(n), np.random.rand(n)
#' twosample.test.s1(x, y, sim_free_null) 
#' 
#' x = (vonmises.rvs(kappa=1, loc=np.pi, size=50) % (2 * np.pi)) / (2 * np.pi)
#' y = (vonmises.rvs(kappa=0, loc=np.pi, size=50) % (2 * np.pi)) / (2 * np.pi)
#' twosample.test.s1(x, y, sim_free_null) 
#' 
#' @export

import numpy as np
from types import SimpleNamespace
import stat_s1_python
import sim_null_stat_python

def twosample_test_s1(x, y, sim_null = None, NR = 500, NC = 1, n = 30):

    if sim_null is None:
        print('No null distribution given as an argument. Simulating with default parameters...')
        sim_null = sim_null_stat_python.sim_null_stat(NR, NC, n)
        print('Done')

    statistic = stat_s1_python.stat_s1(x, y) # Statistic on [0,1)
    pv = np.mean(np.array(sim_null) > statistic) # p-value

    return SimpleNamespace(stat=statistic, pvalue=pv)