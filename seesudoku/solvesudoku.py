import numpy as np


def solve_sudoku(arr):
    arr_copy = np.copy(arr)
    _solve_sudoku_r(arr_copy, 0)
    return arr_copy


def _solve_sudoku_r(arr, p):   
    
    # Get i,j
    def get_loc(p_):
        i = int(p_) / int(9)
        j = p_ % 9
        return i, j
    
    def get_next_pos(p_):   
        try:
            i, j = get_loc(p_)
            while not arr[i, j] == -1:
                p_ += 1   
                i, j = get_loc(p_)
            return p_
        except IndexError:
            return 81    
    
    p_next = get_next_pos(p)         
    # Base Case
    if p > 80 and _is_done(arr):
        return True    
    
    # Recursive Case
    i, j = get_loc(p_next)
    for v in range(1, 10):        
        arr[i, j] = v
        if _is_valid_state(arr):
            if _solve_sudoku_r(arr, get_next_pos(p_next)):
                return True       
    
    arr[i, j] = -1
    return False

def _is_done(arr):
    if -1 in arr.flatten():
        return False
    
    # No '-1's in the board at this point
    # Check rows
    for i in range(9):
        row = arr[i,:]
        if not len(np.unique(row)) == 9:
            return False           

    # Check columns
    for j in range(9):
        col = arr[:,j]
        if not len(np.unique(col)) == 9:
            return False

    # Check 3x3 blocks
    for i in range(3):
        for j in range(3):
            block = arr[i*3:i*3+3, j*3:j*3+3]
            if not len(np.unique(block.flatten())) == 9:
                return False    
    return True
            

def _is_valid_state(arr):
    # Check rows
    for i in range(9):
        row = arr[i,:]
        row = row[np.invert(row == -1)]  # Get rid of '-1's
        if not len(np.unique(row)) == len(row):
            return False
    
    # Check columns
    for j in range(9):
        col = arr[:,j]
        col = col[np.invert(col == -1)]  # Get rid of '-1's
        if not len(np.unique(col)) == len(col):  # If there are duplicates
            return False
    
    # Check 3x3 blocks
    for i in range(3):
        for j in range(3):
            block = arr[i*3:i*3+3, j*3:j*3+3].flatten()
            block = block[np.invert(block == -1)]
            if not len(np.unique(block)) == len(block):  # If there are duplicates
                return False
    
    return True


