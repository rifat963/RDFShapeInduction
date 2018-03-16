#Zero Rule
 ZeroR <- function(X, targetId) {
   # ZeroR Algorithm: Finds the most commonly occuring class
     # 
     # Args:
     # X: data frame or Matrix
     # targetId: response/outcome/target/class feature column number
     
     # Returns:
     # A vector containing the commonly occuring class value and its count 
     if ( is.character(X[, targetId]) | is.factor(X[, targetId]) ) {
       u.x <- unique(X[, targetId])
       u.x.temp <- c()
       for (i in u.x) {
         u.x.temp <- c(u.x.temp, sum(X[, targetId] == i))
       }
       print(u.x.temp)
       names(u.x.temp) <- u.x
       return( c(max(u.x.temp), names(u.x.temp)[which.max(u.x.temp)]) ) 
       }
   return(NULL)
 }
 
 ml_data
 
 ZeroR(ml_data,1)
 
 306/400
 
 
 