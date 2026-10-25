#!/usr/bin/env Rscript
# Optional deterministic clustered-state reference from an installed bkmr0.2.2.
# Set R_LIBS to pinned caller-owned libraries; no fitting or installation occurs.
args <- commandArgs(trailingOnly=TRUE)
if(length(args) != 1L) stop('Usage: Rscript export_cluster_reference.R output.json')
stopifnot(requireNamespace('bkmr',quietly=TRUE),requireNamespace('jsonlite',quietly=TRUE))
stopifnot(as.character(utils::packageVersion('bkmr')) == '0.2.2')
ns <- asNamespace('bkmr')
Z <- matrix(c(-1,.3, -.2,-.4, .5,.7, 1.1,-.6),ncol=2,byrow=TRUE)
X <- cbind(1,c(-.5,.2,.8,1.2))
Znew <- rbind(c(-.3,.1),c(.9,-.2),c(-.3,.1))
cluster <- c(0L,0L,1L,1L)
J <- outer(cluster,cluster,'==')*1
beta <- rbind(c(.2,-.1),c(-.1,.3),c(.4,.05))
r <- rbind(c(.2,0),c(0,.7),c(0,0))
lam <- c(.8,1.5,.4)
lam_cluster <- c(.7,.2,1.1)
delta <- r > 0
yg <- c(-.4,.2,1.3,.7)
yb <- c(0,1,1,0)
ystar <- rbind(c(-.4,.8,1.2,-.2),c(-1,.3,.6,-.8),c(-.2,1.1,.4,-.6))
data.comps <- list(knots=NULL,nlambda=2,randint=TRUE,crossTT=J)
make_case <- function(family) {
  y <- if(family=='gaussian') yg else yb
  sigma <- if(family=='gaussian') c(.6,1.2,.9) else rep(1,3)
  fit <- list(y=y,Z=Z,X=X,beta=beta,r=r,lambda=cbind(lam,lam_cluster),
    sigsq.eps=sigma,delta=delta,ystar=ystar,iter=3L,nsamp=3L,family=family,
    varsel=TRUE,data.comps=data.comps,randint=TRUE,id=cluster+1L)
  class(fit) <- 'bkmrfit'
  exact <- bkmr::ComputePostmeanHnew(fit,Znew=Znew,method='exact',sel=1:3)
  states <- lapply(1:3,function(i) {
    single <- bkmr::ComputePostmeanHnew(fit,Znew=Znew,method='exact',sel=c(i,i))
    vc <- get('makeVcomps',ns)(r[i,],c(lam[i],lam_cluster[i]),Z,data.comps)
    list(K=exp(-get('makeKpart',ns)(r[i,],Z)),Vinv=vc$Vinv,
      logdetVinv=vc$logdetVinv,mean=single$postmean,covariance=single$postvar)
  })
  pips <- bkmr::ExtractPIPs(fit,sel=1:3,z.names=c('z1','z2'))
  fit$groups <- c(1,1)
  grouped <- bkmr::ExtractPIPs(fit,sel=1:3,z.names=c('z1','z2'))
  list(family=family,y=y,Z=Z,X=X,Znew=Znew,cluster_codes=cluster,
    beta=beta,r=r,lambda=lam,lambda_cluster=lam_cluster,sigma2=sigma,
    delta=unname(delta),ystar=if(family=='binomial') ystar else NULL,
    states=states,exact=list(mean=exact$postmean,covariance=exact$postvar),
    pips=pips,grouped_pips=grouped)
}
packages <- c('bkmr','jsonlite','fields','tmvtnorm','truncnorm')
origin <- lapply(packages,function(pkg) list(package=pkg,
  version=as.character(utils::packageVersion(pkg)),package_directory=basename(find.package(pkg))))
script <- sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE))
output <- list(schema_version=1,provenance=list(reference='installed bkmr 0.2.2',
  fixture_kind='constructed clustered bkmrfit objects; no MCMC',R_version=R.version.string,
  packages=origin,exporter_md5=unname(tools::md5sum(script)),
  covariance_convention='exact aggregate uses ddof=1; per-state uses repeated identical sel',
  public_calls=c('ComputePostmeanHnew(method=exact)','ExtractPIPs'),
  cluster_prediction_limit=paste('ComputePostmeanHnew accepts Znew but no new cluster identities;',
    'its returned exposure-function h excludes predicted random intercepts.',
    'Known/unseen h+u and intercept-only prediction need independent block-normal checks.')),
  cases=list(make_case('gaussian'),make_case('binomial')))
jsonlite::write_json(output,args[1],auto_unbox=TRUE,pretty=TRUE,digits=17,na='null',null='null')
cat('Exported deterministic clustered fixtures; no fits or MCMC.\n')
