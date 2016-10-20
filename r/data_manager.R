###########################
# File: data_manager.R
# Description: Downloads football stats and processes them to construct the inputs for a DNN model.
# Date: 10/2016
# Author: Harrison Strowd (harrison@strowd.com)
# Notes:
# To do:
###########################

# Load libraries
library( "optparse" )
library( "httr" )
library( "XML" )
library( "stringr" )
library( "ggplot2" )
library( "plyr" )


# Set application constants.
RAW_DATA_FILE <- "raw_data.csv"
TRAINING_DATA_FILE <- "model_data.train.csv"
TESTING_DATA_FILE <- "model_data.test.csv"

LOAD_DATA <- TRUE
VERBOSE <- TRUE

# Process command line arguments
optParser <- OptionParser()
optParser <- add_option(optParser, c("-v", "--verbose"), action="store_true", default=FALSE,
                        help="Prints additional debugging info [default %default]")
optParser <- add_option(optParser, c("-s", "--skip-load"), action="store_true", default=FALSE,
                        help="Skips loading the raw data [default %default]", dest="skipLoad")
opt = parse_args( optParser )

LOAD_DATA <- ( !opt$skipLoad )
VERBOSE <- opt$verbose

combinedStats <- NULL
if ( LOAD_DATA == TRUE ) {
  cat( "Loading raw data...\n" )

  # Download running back stats for each week
  cafile <- system.file( "CurlSSL", "cacert.pem", package = "RCurl" )

  # Combine all weekly stats into a single list
  combinedStats <- c()
  columnNames <- c(
    "Rank","ID","Player","Position","Week","Team","Opp",
    "RushAtt","RushYds","RushYds.RushAtt","RushTD",
    "Targets","Rec","PassYds","PassTD",
    "Fum","Lost","FantasyPoints"
  )
  for ( week in 1:15 ) {
    if ( VERBOSE ) { cat( paste( "Loading week", week, "data.\n" ) ) }
    query <- paste( c("fs=0&stype=0&sn=1&scope=1&w=", (week - 1), "&ew=", (week - 1), "&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4" ),
                    collapse="" )
    page <- GET(
      "https://fantasydata.com/",
      path="nfl-stats/nfl-fantasy-football-stats.aspx",
      query=query,
      config( cainfo = cafile )
    )
    htmlContent <- readHTMLTable( content( page, as='text' ) )
    weekStatsList <- htmlContent$StatsGrid

    # Standardize column names
    names( weekStatsList ) <- columnNames

    # Convert to a data frame
    weekStats <- as.data.frame( weekStatsList )

    # Convert data to the correct types.
    weekStats$Rank            <- as.numeric( as.character( weekStats$Rank ) )
    weekStats$ID              <- as.numeric( as.character( weekStats$ID ) )
    weekStats$Week            <- as.numeric( as.character( weekStats$Week ) )
    weekStats$RushAtt         <- as.numeric( as.character( weekStats$RushAtt ) )
    weekStats$RushYds         <- as.numeric( as.character( weekStats$RushYds ) )
    weekStats$RushYds.RushAtt <- as.numeric( as.character( weekStats$RushYds.RushAtt ) )
    weekStats$RushTD          <- as.numeric( as.character( weekStats$RushTD ) )
    weekStats$Targets         <- as.numeric( as.character( weekStats$Targets ) )
    weekStats$Rec             <- as.numeric( as.character( weekStats$Rec ) )
    weekStats$PassYds         <- as.numeric( as.character( weekStats$PassYds ) )
    weekStats$PassTD          <- as.numeric( as.character( weekStats$PassTD ) )
    weekStats$Fum             <- as.numeric( as.character( weekStats$Fum ) )
    weekStats$Lost            <- as.numeric( as.character( weekStats$Lost ) )
    weekStats$FantasyPoints   <- as.numeric( as.character( weekStats$FantasyPoints ) )

    # Merge stats across weeks
    combinedStats <- rbind( combinedStats, weekStats )
  }

  # Save off the raw data into a CSV file.
  write.csv( combinedStats, RAW_DATA_FILE, row.names=FALSE )

  cat( "Raw data written to:", RAW_DATA_FILE, "\n" )
} else {
  cat( "Skipping loading the raw data...\n" )
}


if ( is.null( combinedStats ) ) {
  cat( "Loading caw data from:", RAW_DATA_FILE, "\n" )
  combinedStats = read.csv( RAW_DATA_FILE )
}


if ( VERBOSE ) {
  cat( "Pre-processed data structure:\n" )
  cat( str( combinedStats ) )
}



trainingStats <- c()
testingStats <- c()
for ( row in 1:nrow(combinedStats) ) {
  stats = combinedStats[row,]

  playerID <- stats$ID
  week <- stats$Week

  if ( VERBOSE ) { cat( paste( "Processing week", week, "for player", playerID, "\n" ) ) }

  priorWeeks <- subset( combinedStats, combinedStats$ID==playerID & combinedStats$Week>=(week - 2) & combinedStats$Week<=week )
  nextWeek   <- subset( combinedStats, combinedStats$ID==playerID & combinedStats$Week==week )

  if ( nrow( priorWeeks ) != 3 && nrow( nextWeek ) != 1 ) {
    if ( VERBOSE ) { cat( paste( "Skipping stats due to insufficient context. Prior Weeks:", nrow( priorWeeks ), "; Next Week:", nrow( nextWeek ) ) ) }
    next
  }

  avgAttempts       <- ( sum( priorWeeks[, c('RushAtt', 'Targets') ] ) ) / 3
  avgYards          <- ( sum( priorWeeks[, c('RushYds', 'PassYds') ] ) ) / 3
  avgTDs            <- ( sum( priorWeeks[, c('RushTD', 'PassTD') ] ) ) / 3
  avgPoints         <- ( sum( priorWeeks[, c('FantasyPoints') ] ) ) / 3
  lastRushAtt       <- stats$RushAtt
  lastRushYds       <- stats$RushYds
  lastRushTD        <- stats$RushTD
  lastTargets       <- stats$Targets
  lastPassYds       <- stats$PassYds
  lastPassTD        <- stats$PassTD
  lastFantasyPoints <- stats$FantasyPoints
  performanceClass  <- min( floor( nextWeek$FantasyPoints / 5 ), 4 )

  modelStats <- data.frame( avgAttempts, avgYards, avgTDs, avgPoints,
                            lastRushAtt, lastRushYds, lastRushTD,
                            lastTargets, lastPassYds, lastPassTD,
                            lastFantasyPoints, performanceClass )

  if ( week != 15 ) {
    trainingStats <- rbind( trainingStats, modelStats )
  } else {
    testingStats <- rbind( testingStats, modelStats )
  }
}


# Save the training and testing data sets into a CSV file.
cat( paste( "Writing training and testing data sets to", TRAINING_DATA_FILE, "and", TESTING_DATA_FILE, "respectively.\n" ) )

write.csv( trainingStats, TRAINING_DATA_FILE )
write.csv( testingStats, TESTING_DATA_FILE )
