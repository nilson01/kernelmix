#!/usr/bin/env Rscript
# Check that segment boundaries preserve installed-R sampler states and RNG.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('Usage: Rscript check_r_continuation.R config-directory report.json')
script <- sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE)[1L])
source(file.path(dirname(script),'export_fitted_reference.R'))
stopifnot(as.character(utils::packageVersion('bkmr'))=='0.2.2')
reports <- list()
for(case in c('gaussian','gaussian-cluster','binomial','binomial-cluster')) {
  cfg <- jsonlite::read_json(file.path(args[1L],paste0(case,'.json')),simplifyVector=TRUE)
  cfg$warmup <- 0L; cfg$draws <- 40L; cfg$r_draws <- NULL
  control <- list(r.prior='gamma',mu.r=1,sigma.r=sqrt(.5),
    mu.lambda=rep(1,if(cfg$clustered) 2 else 1),
    sigma.lambda=rep(sqrt(.5),if(cfg$clustered) 2 else 1),
    lambda.jump=rep(.6,if(cfg$clustered) 2 else 1),a.sigsq=3,b.sigsq=1,r.jump=.6)
  init <- list(beta=c(0,0),r=c(.8,.8),lambda=rep(.8,if(cfg$clustered) 2 else 1),
    sigsq.eps=.7,ystar=ifelse(cfg$y==1,.5,-.5),h.hat=rep(0,12),delta=c(1,1))
  set.seed(7301L)
  full <- reference_chain(cfg,init,control,segment_size=40L)
  rng_full <- .Random.seed
  set.seed(7301L)
  segmented <- reference_chain(cfg,init,control,segment_size=20L)
  rng_segmented <- .Random.seed
  equal <- vapply(names(full),function(name) identical(full[[name]],segmented[[name]]),logical(1))
  negative <- NULL
  if(cfg$family=='binomial') {
    set.seed(7301L)
    wrong <- reference_chain(cfg,init,control,segment_size=20L,omit_ystar=TRUE)
    negative <- !identical(wrong$ystar,full$ystar)
  }
  reports[[case]] <- list(state_identity=as.list(equal),rng_identity=identical(rng_full,rng_segmented),
    omitted_ystar_rejected=negative,transitions=40L,segments=2L)
  stopifnot(all(equal),identical(rng_full,rng_segmented),is.null(negative)||negative)
}
jsonlite::write_json(reports,args[2L],auto_unbox=TRUE,pretty=TRUE)
cat('All four continuation checks and both omitted-state negative controls passed.\n')
