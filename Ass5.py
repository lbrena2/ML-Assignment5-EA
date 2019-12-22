import matplotlib.pyplot as plt
import numpy as np

#FIX-ME: ASK are this funct and the visualize_rastrigin implied in the rest of the ass or not?
#not general, only for R^2 -> R
def visualize_sphere(x,y):
    X,Y = np.meshgrid(x, y)
    Z = (X**2 + Y**2)
    #plt.contour(X, Y, Z)
    #data = np.dstack((x,y))
    #plt.scatter(x,y)
    plt.show()

#FIX-ME
def visualize_rastrigin(x,y):
    X,Y = np.meshgrid(x, y)
    #Z = rastrigin_test(X,Y)
    #plt.contour(X, Y, Z)
    plt.show()

#input of n dimension
def sphere_test(data):
    #concatenate the input into dataset
    #dataset = np.empty([len(data), data[0].shape[0]])
    #for i, coordinate in enumerate(data):
    #    dataset[i, :] = coordinate
    z = np.sum(data**2, axis = -1)
    return z 

#FIX-ME
def rastrigin_test(data):
    a = np.sum((data**2 - 10 * np.cos(2 * np.pi * data)), axis=1)
    z = 10*data.shape[0] + a
    return z

# in this case the domain can be of arbitrary length
def cem(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function):
    i = 0
    best_fitness_samples = np.zeros((number_of_generations, number_of_features))
    worst_fitness_samples =  np.zeros((number_of_generations, number_of_features))
    while(i < number_of_generations):
        samples = np.random.normal(mean, variance, (number_of_samples, number_of_features))
        #evaluate using target function (the minus is for descending sorting)
        #TIP: if something doesnt work as expected take a look to this sort
        idx = np.argsort(objective_function(samples))
        #select elite based on elite_set and refit gaussian
        mean = np.mean(samples[idx][:elite_set], axis=0)
        #TIP: to avoid variance vanishing add a costant
        variance = np.std(samples[idx][:elite_set], axis=0)
        #variance += 0.1
        #For plots
        best_fitness_samples[i] = samples[idx[0]]
        worst_fitness_samples[i] = samples[idx[-1]]
        #Plot in 2 dimensions
        coord = np.linspace(-5,5,100)
        X,Y = np.meshgrid(coord,coord)
        Z = objective_function(np.dstack([X,Y]))
        plt.figure(1)
        plt.clf()
        plt.contour(X, Y, Z)
        plt.plot(samples[:, 0], samples[:, 1], 'ko')
        plt.pause(1)

        i += 1
    return best_fitness_samples, worst_fitness_samples

def cem_d(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function, number_of_runs):
    best_fitness_samples = np.zeros((number_of_generations, number_of_features))
    worst_fitness_samples =  np.zeros((number_of_generations, number_of_features))
    best_fitness = np.zeros(number_of_generations)
    worst_fitness =  np.zeros(number_of_generations)
    for _ in range(number_of_runs):
        best_fitness_samples, worst_fitness_samples = cem(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function)
        best_fitness += sphere_test(best_fitness_samples)
        worst_fitness += sphere_test(worst_fitness_samples)
    #CHECK: is the order of the operation (sum, mean, sphere_test) correct?
    plt.plot(np.arange(0,number_of_generations), best_fitness / number_of_runs, "r", np.arange(0,number_of_generations), worst_fitness/number_of_runs)
    plt.show()

def nes(mean, variance, number_of_generations, number_of_samples, number_of_features, objective_function, learning_rate):
    i = 0
    best_fitness_samples = np.zeros((number_of_generations, number_of_features))
    worst_fitness_samples =  np.zeros((number_of_generations, number_of_features))
    log_derivatives_mean = np.zeros((number_of_samples, number_of_features))
    log_derivatives_variance = np.zeros((number_of_samples, number_of_features))
    
    while(i < number_of_generations):
        samples = np.random.normal(mean, variance, (number_of_samples, number_of_features))
        evaluated_samples = objective_function(samples)
        for j in range(number_of_samples):
            log_derivatives_mean[j] = (samples[j] - mean) / variance**2
            log_derivatives_variance[j] = ((samples[j] - mean)**2 - variance**2) / variance**3
        log_derivatives_mean_f = log_derivatives_mean * evaluated_samples[..., np.newaxis]
        log_derivatives_variance_f = log_derivatives_variance * evaluated_samples[..., np.newaxis]
        j_derivatives_mean = np.sum(log_derivatives_mean_f  / number_of_samples, axis=0)
        j_derivatives_variance = np.sum(log_derivatives_variance_f  / number_of_samples, axis=0)

        outer_prod_mean = np.zeros((number_of_features, number_of_features, number_of_samples))
        for k,sample in enumerate(log_derivatives_mean):
            outer_prod_mean[:,:,k] = np.outer(sample,sample)
        F_mean = np.sum(outer_prod_mean, axis=-1) / number_of_samples

        outer_prod_variance = np.zeros((number_of_features, number_of_features, number_of_samples))
        for k,sample in enumerate(log_derivatives_variance):
            outer_prod_variance[:,:,k] = np.outer(sample,sample)
        F_variance = np.sum(outer_prod_variance, axis=-1) / number_of_samples
        
        
        mean = mean - learning_rate * (np.matmul(np.linalg.inv(F_mean),j_derivatives_mean))
        variance = variance - learning_rate * (np.matmul(np.linalg.inv(F_variance),j_derivatives_variance))

        #For plots
        idx = np.argsort(- objective_function(samples))
        best_fitness_samples[i] = samples[idx[0]]
        worst_fitness_samples[i] = samples[idx[-1]]
         #Plot in 2 dimensions
        coord = np.linspace(-5,5,100)
        X,Y = np.meshgrid(coord,coord)
        Z = objective_function(np.dstack([X,Y]))
        plt.figure(1)
        plt.clf()
        plt.contour(X, Y, Z)
        plt.plot(samples[:, 0], samples[:, 1], 'ko')
        plt.pause(0.1)
        i += 1
    return best_fitness_samples, worst_fitness_samples

def nes_d(mean, variance, number_of_generations, number_of_samples, number_of_features, objective_function, learning_rate, number_of_runs):
    best_fitness_samples = np.zeros((number_of_generations, number_of_features))
    worst_fitness_samples =  np.zeros((number_of_generations, number_of_features))
    best_fitness = np.zeros(number_of_generations)
    worst_fitness =  np.zeros(number_of_generations)
    for _ in range(number_of_runs):
        best_fitness_samples, worst_fitness_samples = nes(mean, variance, number_of_generations, number_of_samples, number_of_features, objective_function, learning_rate)
        best_fitness += sphere_test(best_fitness_samples)
        worst_fitness += sphere_test(worst_fitness_samples)
    #CHECK: is the order of the operation (sum, mean, sphere_test) correct?
    plt.plot(np.arange(0,number_of_generations), best_fitness / number_of_runs, "r", np.arange(0,number_of_generations), worst_fitness/number_of_runs)
    plt.show()

def cmaes(mean, covariance_matrix, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function):
    i = 0
    best_fitness_samples = np.zeros((number_of_generations, number_of_features))
    worst_fitness_samples =  np.zeros((number_of_generations, number_of_features))
    while(i < number_of_generations):
        #CHECK: just one dimension, is it fine?
        #CHECK: after the first iteration the covariance is no more positive-semidefinite
        samples = np.random.multivariate_normal(mean, covariance_matrix, (number_of_samples))
        # Remember without minus is minimizing
        idx = np.argsort(objective_function(samples))
        # Update covariance first avoids this backup
        old_mean = mean
        mean = np.mean(samples[idx][:elite_set], axis=0)
        covariance_matrix = np.zeros((number_of_features, number_of_features))
        for ii in range(number_of_features):
            for j in range(number_of_features):
                for sample in samples[idx][:elite_set]:
                    covariance_matrix[ii,j] += (sample[ii] - old_mean[ii])*(sample[j] - old_mean[j])
        covariance_matrix = covariance_matrix / elite_set
        #For plots
        best_fitness_samples[i] = samples[idx[0]]
        worst_fitness_samples[i] = samples[idx[-1]]

        #Plot in 2 dimensions
        coord = np.linspace(-5,5,100)
        X,Y = np.meshgrid(coord,coord)
        Z = objective_function(np.dstack([X,Y]))
        plt.figure(1)
        plt.clf()
        plt.contour(X, Y, Z)
        plt.plot(samples[:, 0], samples[:, 1], 'ko')
        plt.pause(0.1)
        
        i += 1
    return best_fitness_samples, worst_fitness_samples

def cmaes_d(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function, number_of_runs):
    best_fitness_samples = np.zeros((number_of_generations, number_of_features))
    worst_fitness_samples =  np.zeros((number_of_generations, number_of_features))
    best_fitness = np.zeros(number_of_generations)
    worst_fitness =  np.zeros(number_of_generations)
    for _ in range(number_of_runs):
        best_fitness_samples, worst_fitness_samples = cmaes(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function)
        best_fitness += objective_function(best_fitness_samples)
        worst_fitness += objective_function(worst_fitness_samples)
    #CHECK: is the order of the operation (sum, mean, sphere_test) correct?
    plt.plot(np.arange(0,number_of_generations), best_fitness / number_of_runs, "r", np.arange(0,number_of_generations), worst_fitness/number_of_runs)
    plt.show()
        
        






### Sphere test plot, change with np.random.normal()
x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
#visualize_sphere(x,y)
#g = np.linspace(-5,5,100)

### cem test 
number_of_generations = 1000
number_of_samples = 100
number_of_features = 2
elite_set = 20
objective_function = sphere_test
number_of_runs = 3
mean = np.random.uniform(-5,5,number_of_features)
variance = np.random.uniform(0,5,number_of_features)
learning_rate = 0.000001

#covariance matrix. CHECK: is correct to use normal?
covariance_matrix = np.diag(np.random.uniform(-5,5,number_of_features))
#covariance_matrix = np.dot(rnd,np.transpose(rnd))


#cem(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function)
#cem_d(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function, number_of_runs)
#nes_d(mean, variance, number_of_generations, number_of_samples, number_of_features, objective_function, learning_rate, number_of_runs)
#cmaes_d(mean, covariance_matrix, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function, number_of_runs)
#cmaes(mean, covariance_matrix, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function)
nes(mean, variance, number_of_generations, number_of_samples, number_of_features, objective_function, learning_rate)


plt.show()

## It's not cheating:
# - Use tensor flow, pythorch for the gradient calc, either use guthub but i have to cite the source 


