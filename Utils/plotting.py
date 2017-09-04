import matplotlib.pyplot as plt

def plotfunc(X,xlbl='X',ylbl='Y',title='TEST'):
    plt.figure()
    plt.plot(X)
    # plt.legend(loc='best')
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(title)
    plt.show()

def plot2D(X,Y,xlbl='X',ylbl='Y',title='TEST'):
    plt.figure()
    plt.plot(X,Y,'r.')
    # plt.legend(loc='best')
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(title)
    plt.show()

def plot2(X,xlbl='X',ylbl='Y',title='TEST'):
    plt.figure()
    plt.plot(X[:,0],X[:,1],'r.')
    # plt.legend(loc='best')
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(title)
    plt.show()
