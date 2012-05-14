#import datetime
\#  timestamp: $datetime.datetime.now().strftime('%c')
\#  Author: $author
\#  Description: $description

\########################################################################
\# Install public header files
\########################################################################
install(FILES
    ${prefix}_api.h
    ${prefix}_${title}_${IOType}.h
    DESTINATION include/${prefix}
)