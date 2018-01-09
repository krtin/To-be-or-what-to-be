# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 5, length = 100, fill = '█', time_string = ''):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    #f = open('results/stats.txt', 'w')
    print('\r%s |%s| %s%% %s (%s) est %s' % (prefix, bar, percent, suffix, str(iteration), time_string), end = '\r')
    #f.write(prefix +' | '+ bar +'| '+ percent +'% '+ suffix + '(' + str(iteration) + ')')
    #f.close()
    # Print New Line on Complete
    if iteration == total:
        print()
