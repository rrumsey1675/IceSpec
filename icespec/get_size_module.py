# rumz 5/31/24

### Feb 2025
### NOT IN USE

import pandas as pd

def retrieveData(DataFrame, Row):

    TopPx = DataFrame._get_value(Row, 'TopPx') # getting Core Top line
    BottomPx = DataFrame._get_value(Row, 'BottomPX') # getting Core Bottom line
    LeftPx = DataFrame._get_value(Row, 'LeftPX') # getting Core Left Pixel
    RightPx = DataFrame._get_value(Row, 'RightPx') # getting Core Right pixel

    dataSize = [TopPx, BottomPx, LeftPx, RightPx]

    Standard1TopPx = DataFrame._get_value(Row, 'Standard1 TopPx')
    Standard1BottomPx = DataFrame._get_value(Row, 'Standard1 BottomPx')
    Standard1LeftPx = DataFrame._get_value(Row, 'Standards LeftPx')
    Standard1RightPx = DataFrame._get_value(Row, 'Standards RightPx')

    stdSize = [Standard1TopPx, Standard1BottomPx, Standard1LeftPx, Standard1RightPx]

    return dataSize, stdSize