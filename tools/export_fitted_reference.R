#!/usr/bin/env Rscript
# Optional full-chain comparison against installed bkmr, outside Python runtime.
# Retain the R transition and RNG stream while bounding diagnostic allocation.
reference_chain <- function(cfg, init, control, segment_size=NULL, omit_ystar=FALSE) {
  if (is.null(segment_size)) segment_size <- cfg$r_segment_size
  ndraws <- if(is.null(cfg$r_draws)) cfg$draws else cfg$r_draws
  remaining <- cfg$warmup+ndraws
  if (is.null(segment_size)) segment_size <- remaining
  stopifnot(segment_size >= 1L)
  blocks <- list()
  while (remaining > 0L) {
    transitions <- min(segment_size,remaining)
    fit <- bkmr::kmbayes(y=as.numeric(cfg$y),Z=as.matrix(cfg$Z),X=as.matrix(cfg$X),
      family=cfg$family,id=if(isTRUE(cfg$clustered)) as.numeric(cfg$cluster) else NULL,
      iter=transitions+1L,verbose=FALSE,varsel=FALSE,est.h=FALSE,
      starting.values=init,control.params=control,
      Znew=as.matrix(cfg$Z)[c(1,6,12),,drop=FALSE])
    last <- transitions+1L
    state_names <- c('beta','r','lambda','sigsq.eps','h.hat','delta',
                     if(cfg$family=='binomial') 'ystar' else NULL)
    init <- lapply(fit[state_names],function(value) {
      if(is.null(dim(value))) value[last] else as.numeric(value[last,])
    })
    if(omit_ystar) init$ystar <- ifelse(cfg$y==1,.5,-.5)
    sample_names <- c(state_names,'hnew')
    blocks[[length(blocks)+1L]] <- lapply(fit[sample_names],function(value) {
      if(is.null(dim(value))) matrix(value[-1L],ncol=1L) else value[-1L,,drop=FALSE]
    })
    remaining <- remaining-transitions
  }
  combined <- lapply(names(blocks[[1L]]),function(name) {
    do.call(rbind,lapply(blocks,function(block) block[[name]]))
  })
  names(combined) <- names(blocks[[1L]])
  # Add one unused row so existing warmup indexing stays explicit.
  lapply(combined,function(value) rbind(NA_real_,value))
}

main <- function() {
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop('Usage: Rscript export_fitted_reference.R config.json output-directory')
stopifnot(requireNamespace('bkmr',quietly=TRUE),requireNamespace('jsonlite',quietly=TRUE))
stopifnot(as.character(utils::packageVersion('bkmr')) == '0.2.2')
cfg <- jsonlite::read_json(args[1],simplifyVector=TRUE)
dir.create(args[2],recursive=TRUE,showWarnings=FALSE)
Z <- as.matrix(cfg$Z); X <- as.matrix(cfg$X)
y <- as.numeric(cfg$y)
stopifnot(nrow(Z)==length(y),nrow(X)==length(y),all(is.finite(Z)),all(is.finite(X)))
clustered <- isTRUE(cfg$clustered)
ndraws <- if(is.null(cfg$r_draws)) cfg$draws else cfg$r_draws
control <- list(r.prior='gamma',mu.r=1,sigma.r=sqrt(.5),
                mu.lambda=rep(1,if(clustered) 2 else 1),
                sigma.lambda=rep(sqrt(.5),if(clustered) 2 else 1),
                lambda.jump=rep(.6,if(clustered) 2 else 1),
                a.sigsq=3,b.sigsq=1,r.jump=.6)
started <- proc.time()[['elapsed']]
warnings <- character()
for (ch in seq_len(cfg$chains)) {
  set.seed(as.integer(cfg$r_seeds[ch]))
  init <- list(beta=rep(0,ncol(X)),r=rep(.6+.2*ch,ncol(Z)),
               lambda=rep(.6+.2*ch,if(clustered) 2 else 1),sigsq.eps=.7,
               ystar=ifelse(y==1,.5,-.5),h.hat=rep(0,length(y)))
  fit <- withCallingHandlers(reference_chain(cfg, init, control),warning=function(w) {
    warnings <<- c(warnings,conditionMessage(w));invokeRestart('muffleWarning')
  })
  sel <- seq.int(cfg$warmup+2L,cfg$warmup+ndraws+1L)
  out <- cbind(fit$beta[sel,,drop=FALSE],fit$r[sel,,drop=FALSE],
               fit$lambda[sel,,drop=FALSE])
  colnames(out) <- c('beta_0','beta_1','r_0','r_1','lambda',
                     if(clustered) 'lambda_cluster' else NULL)
  if(cfg$family=='gaussian') out <- cbind(out,sigma2=fit$sigsq.eps[sel])
  h <- fit$hnew[sel,,drop=FALSE]
  out <- cbind(out,h_0=h[,1],h_5=h[,2],h_11=h[,3],h_contrast=h[,3]-h[,1])
  connection <- gzfile(file.path(args[2],paste0('r-chain-',ch,'.csv.gz')),'wt')
  utils::write.csv(out,connection,row.names=FALSE)
  close(connection)
}
jsonlite::write_json(list(reference='installed bkmr',version=as.character(utils::packageVersion('bkmr')),
  R_version=R.version.string,library=normalizePath(find.package('bkmr')),
  call='bkmr::kmbayes with Znew; full posterior chains',
  chains=cfg$chains,warmup=cfg$warmup,draws=ndraws,
  initialization_discarded=TRUE,r_segment_size=cfg$r_segment_size,control=control,warnings=warnings,
  session=capture.output(sessionInfo()),
  loaded_packages=lapply(loadedNamespaces(),function(pkg) list(package=pkg,
    version=as.character(utils::packageVersion(pkg)),library=find.package(pkg))),
  seconds=proc.time()[['elapsed']]-started),file.path(args[2],'r-metadata.json'),
  auto_unbox=TRUE,pretty=TRUE,digits=17)

}
if(sys.nframe()==0L) main()
