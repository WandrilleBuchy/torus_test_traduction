#' Wasserstein distribution-free statistic on S^1
#'
#' Computes Wasserstein statistic between two samples on S^1, parameterized as a periodic [0,1).
#'
#' @param x numerical vector of sample in [0,1).
#' @param y numerical vector of sample in [0,1).
#'
#' @return The statistic realization of samples x, y.
#'
#' @references [1] Delon, J., Salomon, J., Sobolevskii, A.: Fast transport optimization for Monge costs on the circle. SIAM J. Appl. Math. 70(7), 2239–2258 (2010).
#' [2] RAMDAS, A., GARCIA, N. and CUTURI, M. (2015). On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests. Entropy 19.
#'
#' @examples
#' import numpy as np
#' from scipy.stats import vonmises
#'
#' set.seed(10)
#' stat.s1(np.random.rand(50), np.random.rand(50))
#' stat.s1(np.random.rand(50), (vonmises.rvs(kappa=1, loc=np.pi, size=50) % (2 * np.pi)) / (2 * np.pi))
#'
#' @export

from scipy.stats import ecdf
from scipy.integrate import fixed_quad
import numpy as np

def stat_s1(x, y, n_quad = 100):

    fy = ecdf(y)
    
    def h(t):
        return np.array([fy.cdf.evaluate(np.quantile(x, ti)) - ti for ti in t])

    optimal_alpha, _ = fixed_quad(h, 0, 1, n = n_quad)

    def h_alpha(t):
        return np.array([(fy.cdf.evaluate(np.quantile(x, ti)) - ti - optimal_alpha) ** 2 for ti in t])

    n, m = len(x), len(y)
    statistic, _ = fixed_quad(h_alpha, 0, 1, n = n_quad)
    
    return(n*m/(n+m) * statistic)
