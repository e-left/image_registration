import numpy as np

def myLocalDescriptor(I, p, rhom, rhoM, rhostep, N):
    # check to see if all circles are withing the image,
    # radii are positive, and rhoM >= rhom
    (Mdim, Ndim) = I.shape
    min_x = p[0] - rhoM
    max_x = p[0] + rhoM
    min_y = p[1] - rhoM
    max_y = p[1] + rhoM

    if rhom <= 0 or rhoM <= 0 or rhom > rhoM or min_x < 0 or max_x >= Mdim or min_y < 0 or max_y >= Ndim:
        # return np.zeros((int((rhoM - rhom) / rhostep), 1))
        return None

    # generate all possibly radii
    radii = np.arange(rhom, rhoM, rhostep)

    # calculate all xp and store them
    circle_lists = []
    for r in radii:
        points = []
        for n in range(N):
            theta = n * 2 * np.pi / N
            
            # convert to cartesian
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # add origin
            x = x + p[0]
            y = y + p[1]

            # find interpolated pixel value
            low_x = int(np.floor(x))
            high_x = int(np.ceil(x))
            low_y = int(np.floor(y))
            high_y = int(np.ceil(y))
            i_v = I[low_x][low_y] / 4 + I[low_x][high_y] / 4 + I[high_x][low_y] / 4 + I[high_x][high_y] / 4

            # append pixel value to points
            points.append(i_v)
        # append vector to list
        circle_lists.append(points)

    # return average of each xp
    return np.mean(circle_lists, axis=1)

def myLocalDescriptorUpgrade(I, p, rhom, rhoM, rhostep, N):
    # check to see if all circles are withing the image,
    # radii are positive, and rhoM >= rhom
    (Mdim, Ndim) = I.shape
    min_x = p[0] - rhoM
    max_x = p[0] + rhoM
    min_y = p[1] - rhoM
    max_y = p[1] + rhoM

    if rhom <= 0 or rhoM <= 0 or rhom > rhoM or min_x < 0 or max_x >= Mdim or min_y < 0 or max_y >= Ndim:
        return []

    # generate all possibly radii
    radii = np.arange(rhom, rhoM, rhostep)

    # calculate all xp and store them
    circle_lists = []
    for r in radii:
        points = []
        for n in range(N):
            theta = n * 2 * np.pi / N
            
            # convert to cartesian
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # add origin
            x = x + p[0]
            y = y + p[1]

            # find interpolated pixel value
            low_x = int(np.floor(x))
            high_x = int(np.ceil(x))
            low_y = int(np.floor(y))
            high_y = int(np.ceil(y))
            i_v = I[low_x][low_y] / 4 + I[low_x][high_y] / 4 + I[high_x][low_y] / 4 + I[high_x][high_y] / 4

            # append pixel value to points
            points.append(i_v)
        # append vector to list
        circle_lists.append(points)

    # return average of each xp, along with maximum difference
    avg =  np.mean(circle_lists, axis=1)
    diff = np.max(circle_lists, axis=1) - np.min(circle_lists, axis=1)
    return np.array((avg, diff)).T
