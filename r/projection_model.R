###########################
# File: projection_model.R
# Description: Runs a deep nueral net model to predict the performance of fantasy football players.
# Date: 10/2016
# Author: Harrison Strowd (harrison@strowd.com)
# Notes:
# To do:
###########################

# Load libraries
library( "optparse" )
library( "deepnet" )


# Set application Constants.
TRAINING_DATA_FILE <- "model_data.train.csv"
TESTING_DATA_FILE <- "model_data.test.csv"

VERBOSE <- TRUE


# Process command line arguments
opt_parser <- OptionParser()
opt_parser <- add_option(opt_parser, c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Prints additional debugging info [default %default]")
opt = parse_args( opt_parser )

VERBOSE <- opt$verbose


cat( paste( "Loading training and testing data from", TRAINING_DATA_FILE, "and", TESTING_DATA_FILE, "respectively.\n" ) )

trainingData <- read.csv(TRAINING_DATA_FILE)
testingData <- read.csv(TESTING_DATA_FILE)


modelInputs <- c( "avgAttempts", "avgYards", "avgTDs", "avgPoints",
                  "lastRushAtt", "lastRushYds", "lastRushTD",
                  "lastTargets", "lastPassYds", "lastPassTD",
                  "lastFantasyPoints" )
trainingInputs <- as.matrix( trainingData[, modelInputs ] )
trainingResults <- trainingData$performanceClass

testingInputs <- as.matrix( testingData[, modelInputs ] )
testingResults <- testingData$performanceClass


cat( "Training the model...\n" )
if ( VERBOSE ) {
  cat( "Training Inputs: \n" )
  print( trainingInputs )

  cat( "Training Results: \n" )
  print( trainingResults )
}
dnn <- dbn.dnn.train( trainingInputs, trainingResults, hidden=c(10, 20, 40, 20, 10) )


cat( "Testing the model...\n" )
if ( VERBOSE ) {
  cat( "Testing Inputs: \n" )
  print( testingInputs )

  cat( "Testing Results: \n" )
  print( testingResults )
}
nn.test( dnn, testingInputs, testingResults )


predictionInputs <- as.matrix( data.frame( avgAttempts=c( 10.75, 6.23 ), avgYards=c( 120.34, 76.83 ), avgTDs=c( 1.52, 1.10 ), avgPoints=c( 18.63, 9.27 ),
                                           lastRushAtt=c( 5, 3 ), lastRushYds=c( 75, 17 ), lastRushTD=c( 1, 0 ),
                                           lastTargets=c( 2, 12 ), lastPassYds=c( 14, 101 ), lastPassTD=c( 0, 2 ),
                                           lastFantasyPoints=c( 10.63, 19.83 ) ) )

results <- nn.predict( dnn, predictionInputs )

cat( "Testing the model...\n" )
if ( VERBOSE ) {
  cat( "Prediction Inputs: \n" )
  print( predictionInputs )

  cat( "Prediction Results:\n" )
  print( results )
}
