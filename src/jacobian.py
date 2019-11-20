def jacobian(a, b, c, d):
    """ a, b, c, d are the thetas for each link """
    return np.matrix([
        [-3*np.cos(b)*np.cos(c)*np.sin(a)-3*np.cos(a)*np.sin(c)+2*np.cos(d)*(-np.cos(b)*np.cos(c)*np.sin(a)-np.cos(a)*np.sin(c))+2*np.sin(a)*np.sin(b)*np.sin(d),
            -3*np.cos(a)*np.cos(c)*np.sin(b)-2*np.cos(a)*np.cos(c)*np.cos(d)*np.sin(b)-2*np.cos(a)*np.cos(b)*np.sin(d),
            -3*np.cos(c)*np.sin(a)-3*np.cos(a)*np.cos(b)*np.sin(c)+2*np.cos(d)*(-np.cos(c)*np.sin(a)-np.cos(a)*np.cos(b)*np.sin(c)),
            -2*np.cos(a)*np.cos(d)*np.sin(b)-2*(np.cos(a)*np.cos(b)*np.cos(c)-np.sin(a)*np.sin(c))*np.sin(d)],
        [3*np.cos(a)*np.cos(b)*np.cos(c)-3*np.sin(a)*np.sin(c)+2*np.cos(d)*(np.cos(a)*np.cos(b)*np.cos(c)-np.sin(a)*np.sin(c))-2*np.cos(a)*np.sin(b)*np.sin(d),
            -3*np.cos(c)*np.sin(a)*np.sin(b)-2*np.cos(c)*np.cos(d)*np.sin(a)*np.sin(b)-2*np.cos(b)*np.sin(a)*np.sin(d),
            3*np.cos(a)*np.cos(c)-3*np.cos(b)*np.sin(a)*np.sin(c)+2*np.cos(d)*(np.cos(a)*np.cos(c)-np.cos(b)*np.sin(a)*np.sin(c)),
            -2*np.cos(d)*np.sin(a)*np.sin(b)-2*(np.cos(b)*np.cos(c)*np.sin(a)+np.cos(a)*np.sin(c))*np.sin(d)],
        [0,
            -3*np.cos(b)*np.cos(c)-2*np.cos(b)*np.cos(c)*np.cos(d)+2*np.sin(b)*np.sin(d),
            3*np.sin(b)*np.sin(c)+2*np.cos(d)*np.sin(b)*np.sin(c),
            -2*np.cos(b)*np.cos(d)+2*np.cos(c)*np.sin(b)*np.sin(d)],
        [0,0,1,0],
        [0,1,0,1],
        [1,0,0,0]])
