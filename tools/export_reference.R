#!/usr/bin/env Rscript
# Optional installed-bkmr exporter. Run with R_LIBS pointing to a pinned library.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop('Usage: Rscript export_reference.R output.json')
stopifnot(requireNamespace('bkmr', quietly = TRUE), requireNamespace('jsonlite', quietly = TRUE))
stopifnot(as.character(utils::packageVersion('bkmr')) == '0.2.2')
ns <- asNamespace('bkmr')
Z <- matrix(c(-1,.3, -.2,-.4, .5,.7, 1.1,-.6), ncol=2, byrow=TRUE)
X <- cbind(1,c(-.5,.2,.8,1.2))
Znew <- rbind(c(-.3,.1),c(.9,-.2),c(-.3,.1))
beta <- rbind(c(.2,-.1),c(-.1,.3),c(.4,.05))
r <- rbind(c(.2,0),c(0,.7),c(0,0))
lam <- c(.8,1.5,.4)
delta <- r > 0
yg <- c(-.4,.2,1.3,.7)
yb <- c(0,1,1,0)
ystar <- rbind(c(-.4,.8,1.2,-.2),c(-1,.3,.6,-.8),c(-.2,1.1,.4,-.6))
data.comps <- list(knots=NULL,nlambda=1,randint=FALSE,crossTT=NULL)
case <- function(family) {
  y <- if(family == 'gaussian') yg else yb
  sigma <- if(family == 'gaussian') c(.6,1.2,.9) else rep(1,3)
  fit <- list(y=y,Z=Z,X=X,beta=beta,r=r,lambda=matrix(lam,ncol=1),sigsq.eps=sigma,
              delta=delta,ystar=ystar,iter=3L,nsamp=3L,family=family,varsel=TRUE,data.comps=data.comps)
  class(fit) <- 'bkmrfit'
  exact <- bkmr::ComputePostmeanHnew(fit,Znew=Znew,method='exact',sel=1:3)
  states <- lapply(1:3,function(i) {
    single <- bkmr::ComputePostmeanHnew(fit,Znew=Znew,method='exact',sel=c(i,i))
    vc <- get('makeVcomps',ns)(r[i,],lam[i],Z,data.comps)
    list(K=exp(-get('makeKpart',ns)(r[i,],Z)),Vinv=vc$Vinv,logdetVinv=vc$logdetVinv,
         cross_kernel=exp(-get('makeKpart',ns)(r[i,],Znew,Z)),
         mean=single$postmean,covariance=single$postvar)
  })
  pips <- bkmr::ExtractPIPs(fit,sel=1:3,z.names=c('z1','z2'))
  fit$groups <- c(1,1)
  grouped <- bkmr::ExtractPIPs(fit,sel=1:3,z.names=c('z1','z2'))
  list(family=family,y=y,Z=Z,X=X,Znew=Znew,beta=beta,r=r,lambda=lam,sigma2=sigma,
       delta=unname(delta),ystar=if(family=='binomial') ystar else NULL,
       states=states,exact=list(mean=exact$postmean,covariance=exact$postvar),
       pips=pips,grouped_pips=grouped)
}
prior_cases <- lapply(c('gamma','unif','invunif'),function(kind) {
  pars <- list(mu.r=5,sigma.r=5,r.a=0,r.b=100)
  fun <- get('set.r.MH.functions',ns)(kind)$rprior.logdens
  values <- c(.005,.02,.5,2,99,101)
  dens <- vapply(values,function(v)fun(v,pars),numeric(1))
  list(distribution=kind,values=values,log_density=ifelse(is.finite(dens),dens,NA_real_))
})
script <- sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE))
origin <- lapply(c('bkmr','jsonlite','fields','tmvtnorm','truncnorm'),function(pkg)
  list(package=pkg,version=as.character(utils::packageVersion(pkg)),package_directory=basename(find.package(pkg))))
output <- list(schema_version=1,provenance=list(reference='installed bkmr 0.2.2',
  fixture_kind='constructed bkmrfit objects; no MCMC',R_version=R.version.string,
  packages=origin,exporter_md5=unname(tools::md5sum(script)),
  exact_covariance_convention='R var(conditional means) uses ddof=1; repeated identical sel for per-state moments',
  selection=1:3),cases=list(case('gaussian'),case('binomial')),priors=prior_cases,
  lambda_prior=list(values=c(.1,1,10),log_density=dgamma(c(.1,1,10),shape=1,rate=.1,log=TRUE)))
jsonlite::write_json(output,args[1],auto_unbox=TRUE,pretty=TRUE,digits=17,na='null',null='null')
