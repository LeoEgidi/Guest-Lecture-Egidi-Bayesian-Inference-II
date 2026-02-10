#######################
# Fit using rstan (HMC)
#######################
library(ggplot2)
library(dplyr)
library(rstan)
library(bayesplot)

rstan_options(auto_write = TRUE)

hmm_data <- readRDS("data/hmm_example.RDS")

stan_data <- list(N = length(hmm_data$y),
                  K = 2,
                  y = hmm_data$y)

hmm_fit <- stan("models/hmm_example.stan", data = stan_data, iter = 1e3, chains = 4)
#saveRDS(list(fit = hmm_fit, data = stan_data), "results/hmm_example.RDS")
print(hmm_fit, pars = "z_star", include = FALSE, probs = c(0.05,0.95))

# Marginal posterior plots
mcmc_intervals(hmm_fit, regex_pars = c("mu", "theta"))
mcmc_areas(hmm_fit, regex_pars = c("mu", "theta"))

# collect values from the MCMC samples
samples <- as.matrix(hmm_fit)

psi_indx <- grep("^mu\\[", colnames(samples))
theta_indx <- grep("^theta\\[", colnames(samples))
z_star_indx <- grep("^z_star\\[", colnames(samples))

# trace-plots
traceplot(hmm_fit, pars = "theta")
traceplot(hmm_fit, pars = "mu")
mcmc_trace(as.array(hmm_fit), regex_pars = "^theta\\[|^mu\\[", facet_args = list(nrow = 2))

colMeans(samples[,theta_indx])
colMeans(samples[,psi_indx])

z_star <- colMeans(samples[,z_star_indx])

# visualization
pdf("media/hmm_example.pdf", width = 12, height = 9)
par(mfrow=c(2,1))
plot(hmm_data$z, type="l",
     main = "Latent States",
     ylab = "State Value",
     xlab = "Time")
points(z_star, cex = 0.5)
legend("bottomright", c("Actual","Predicted"), pch = c(NA,1), lty = c(1,NA), cex = 0.8)
plot(hmm_data$y, type = "l",
     main = "Observed Output",
     ylab = "Observation Value",
     xlab = "Time")
y_plt <- hmm_data$y
y_plt[hmm_data$z==1] <- NA
lines(y_plt, lwd = 3)
legend("bottomright", c("State 1","State 2"), lty = c(1,1), lwd = c(1,3), cex = 0.8)
dev.off()

# posterior predictions
# simulate observations for each iteration in the sample
# extract samples
samples <- as.matrix(hmm_fit)
theta <- samples[,grep("^theta",colnames(samples))]
mu <- samples[,grep("^mu",colnames(samples))]
z_star <- samples[,grep("^z_star",colnames(samples))]
y_hat <- list()
for (i in 1:nrow(samples)) {
  psi_seq <- sapply(z_star[i,], function(x){mu[i,x]})
  y_hat[[i]] <- rnorm(length(psi_seq), psi_seq, 1)
}

# plot
indxs <- sample(length(y_hat), 100, replace = FALSE)
plot(hmm_data$y, type = "n",
     main = "Observed vs Predicted Output",
     ylab = "Observation Value",
     xlab = "Time",
     ylim = c(0,11))
for (i in indxs) {
  lines(y_hat[[i]], col = "#ff668890")
}
lines(hmm_data$y, lwd = 2)
legend("bottomright", c("Observed","Predicted"), col = c("#000000","#ff668890"), lty = c(1,1), lwd = c(2,1), cex = 0.8)

# visualization
par(mfrow=c(2,1))
plot(hmm_data$z, type="s",
     main = "Latent States",
     ylab = "State Value",
     xlab = "Time",
     ylim = c(0.5,2.5), yaxt = "n")
axis(2, 1:2, 1:2)
points(colMeans(z_star), cex = 0.5)
legend("bottomright", c("Actual","Predicted"), pch = c(NA,1), lty = c(1,NA), cex = 0.5)
plot(hmm_data$y, type = "l",
     main = "Observed Output",
     ylab = "Observation Value",
     xlab = "Time")
y_plt <- hmm_data$y
y_plt[hmm_data$z==1] <- NA
lines(y_plt, lwd = 3)
legend("bottomright", c("State 1","State 2"), lty = c(1,1), lwd = c(1,3), cex = 0.8)


###################
# Fit using cmdstan
###################


library(cmdstanr)

model = cmdstan_model("models/hmm_example.stan") # model compilation

# The $sample() method for CmdStanModel objects runs Stan's default MCMC algorithm
fit_hmc = model$sample(data = stan_data, parallel_chains = 4) #HMC

# Variational 'pathfinder'
fit_pf = model$pathfinder(data = stan_data, seed = 123,draws = 4000) # VI PathFinder

# Maximum a posteriori
fit_map = model$optimize(data = stan_data, jacobian = TRUE, seed = 123) #MAP

# Maximum likelihood estimate
fit_mle = model$optimize(data = stan_data, seed = 123) #MLE

# Laplace approximation
# The $laplace() method produces a sample from a normal approximation centered at the mode of a distribution in the unconstrained space.
fit_laplace = model$laplace(mode = fit_map, draws = 4000, 
                            data = stan_data, seed = 123, refresh = 1000) #Laplace

# We can run Stan's experimental Automatic Differentiation Variational Inference (ADVI) using the $variational() method. For details on the ADVI algorithm
fit_vb = model$variational(data = stan_data, seed = 123,draws = 4000) # Mean-field

fit_hmc$print(c("theta", "mu"))
fit_pf$print(c("theta", "mu"))
fit_vb$print(c("theta", "mu"))
fit_laplace$print(c("theta", "mu"))
fit_mle$print(c("theta", "mu"))
fit_map$print(c("theta", "mu"))


plot_comparison_hmm_means = function(fit_mcmc, fit_pathfinder, fit_laplace) {
  draws = dplyr::tibble()
  for (par in c("mu[1]", "mu[2]")) {
    draws_mcmcm <- as.data.frame(fit_mcmc$draws(variables = c(par), format = "draws_matrix")) %>%
      `colnames<-`("value") %>%
      mutate(algorithm = "hmc", par_name=par)
    draws_pf <- as.data.frame(fit_pathfinder$draws(variables = c(par), format = "draws_matrix")) %>%
      `colnames<-`("value") %>%
      mutate(algorithm = "pathfinder", par_name=par)
    draws_laplace <- as.data.frame(fit_laplace$draws(variables = c(par), format = "draws_matrix")) %>%
      `colnames<-`("value") %>%
      mutate(algorithm = "laplace", par_name=par)
    draws <- dplyr::bind_rows(draws, draws_mcmcm, draws_pf, draws_laplace)
  }
  
  draws %>%
    ggplot(mapping = aes(x=value, fill=algorithm, col=algorithm)) +
    geom_density(alpha=.5) +
    facet_wrap(~ par_name, scales = "free") +
    theme_bw() +
    scale_fill_manual(values = c("deepskyblue3", "indianred3" , "lightgreen")) +
    scale_color_manual(values = c("deepskyblue3", "indianred3", "lightgreen")) +
    theme(legend.position = "bottom")
}

plot_comparison_hmm_means(fit_hmc, fit_pf, fit_laplace)


