#' Wasserstein statistic null distribution
#'
#' Simulates Wasserstein statistic null distribution
#'
#' @param NR Number of replicas to simulate
#' @param NC Number of cores for parallel computation
#' @param n Sample size for the simulated samples
#'
#' @return A sample of size NR simulating the null distribution of the statistic.
#'
#' @references [1] Delon, J., Salomon, J., Sobolevskii, A.: Fast transport optimization for Monge costs on the circle. SIAM J. Appl. Math. 70(7), 2239–2258 (2010).
#' [2] RAMDAS, A., GARCIA, N. and CUTURI, M. (2015). On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests. Entropy 19.
#'
#' @examples
#' sim.null.stat(100)
#'
#' @export

from scipy.stats import ecdf
from joblib import Parallel, delayed
import numpy as np
from scipy.integrate import quad

def sim_null_stat(NR, NC = 1, n = 30):

    def stat_rep(i, m):
        x, y = np.random.rand(m), np.random.rand(m)
        fy = ecdf(y)
        def h(t):
            return float(fy.cdf.evaluate(np.quantile(x, t)) - t)

        optimal_alpha, _ = quad(h, 0, 1)
        
        def h_alpha(t):
            return(float((fy.cdf.evaluate(np.quantile(x, t)) - t - optimal_alpha))**2)

        n, m = len(x), len(y)
        optimal_h_alpha, _ = quad(h_alpha, 0, 1)
        
        return(n*m/(n+m) * optimal_h_alpha)

    sim = Parallel(n_jobs=NC)(delayed(stat_rep)(i, m=n) for i in range(1, NR+1))

    return(sim)