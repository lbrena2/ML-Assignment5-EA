import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

#FIX-ME: Add colour map, fix axes values
#not general, only for R^2 -> R
def visualize_obj_function(objective_function):
    coord = np.linspace(-5,5,100)
    X,Y = np.meshgrid(coord,coord)
    Z = objective_function(np.dstack([X,Y]))
    plt.figure(1)
    plt.clf()
    contour = plt.contourf(X, Y, Z, 20, cmap='RdGy')
    plt.colorbar()
    # samples = np.random.uniform(-5,5,(100,2))
    # plt.plot(samples[:, 0], samples[:, 1], 'ko')
    plt.show()

def sphere_test(data):
    z = np.sum(data**2, axis = -1)
    return z 

def rastrigin_test(data):
    a = np.sum((data**2 - 10 * np.cos(2 * np.pi * data)), axis=-1)
    z = 10*data.shape[0] + a
    # return (10 * dimension) + np.sum(np.square(samples) - A * np.cos(2 * np.pi * samples))
    return z

def plot_samples_on_contour(samples, objective_function):
    coord = np.linspace(-5,5,100)
    X,Y = np.meshgrid(coord,coord)
    Z = objective_function(np.dstack([X,Y]))
    plt.figure(1)
    plt.clf()
    plt.contour(X, Y, Z)
    plt.plot(samples[:, 0], samples[:, 1], 'ko')
    plt.pause(0.1)


def cem(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function):
    i = 0
    best_fitness_samples = np.zeros((number_of_generations, number_of_features))
    worst_fitness_samples =  np.zeros((number_of_generations, number_of_features))
    while(i < number_of_generations):
        samples = np.random.normal(mean, variance, (number_of_samples, number_of_features))
        #evaluate using target function (the minus is for descending sorting)
        idx = np.argsort(objective_function(samples))
        #select elite based on elite_set and refit gaussian
        mean = np.mean(samples[idx][:elite_set], axis=0)
        #TIP: to avoid variance vanishing add a costant
        variance = np.std(samples[idx][:elite_set], axis=0)
        #variance += 0.001
        #For plots
        best_fitness_samples[i] = samples[idx[0]]
        worst_fitness_samples[i] = samples[idx[-1]]
        #Plot in 2 dimensions
        # plot_samples_on_contour(samples, objective_function)
        i += 1
    # plt.plot(np.arange(0,number_of_generations), objective_function(best_fitness_samples), "r")
    # plt.title('Number of Generations: ' + str(number_of_generations) + '\nPopulation number: ' + str(number_of_samples) + '\nNumber of features: ' + str(number_of_features)\
    #     +'\n Elite set: '+ str(elite_set) + '\n Best fitness: ' +  str(objective_function(best_fitness_samples[-1])))
    # plt.xlabel('Generation')
    # plt.ylabel('Fitness')
    # plt.show()
    return best_fitness_samples, worst_fitness_samples

def cem_d(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function, number_of_runs):
    best_fitness_samples = np.zeros((number_of_generations, number_of_features))
    worst_fitness_samples =  np.zeros((number_of_generations, number_of_features))
    best_fitness = np.zeros(number_of_generations)
    worst_fitness =  np.zeros(number_of_generations)
    for _ in range(number_of_runs):
        best_fitness_samples, worst_fitness_samples = cem(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function)
        best_fitness +=  objective_function(best_fitness_samples)
        worst_fitness +=  objective_function(worst_fitness_samples)
    #prints for the report:
    # print("Avg best fitness:" + str(np.min(best_fitness/number_of_runs)))
    # print("Avg worst fitness:" + str(np.min(worst_fitness/number_of_runs)))
    #CHECK: is the order of the operation (sum, mean, sphere_test) correct?
    plt.plot(np.arange(0,number_of_generations), best_fitness / number_of_runs, "r", label="best fitness")
    plt.plot(np.arange(0,number_of_generations), worst_fitness/number_of_runs, "b", label="worse fitness")
    plt.title('Number of Generations: ' + str(number_of_generations) + '\nPopulation number: ' + str(number_of_samples) + '\nNumber of features: ' + str(number_of_features)\
        +'\n Elite set: '+ str(elite_set))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(loc="upper right")
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
        
        log_derivatives_mean = (samples - mean) / variance**2
        log_derivatives_variance = ((samples - mean)**2 - variance**2) / variance**3
        log_derivatives_mean_f = evaluated_samples[..., np.newaxis] * log_derivatives_mean
        log_derivatives_variance_f = evaluated_samples[..., np.newaxis] * log_derivatives_variance

        j_derivatives_mean = np.sum(log_derivatives_mean_f, axis=0) / number_of_samples
        j_derivatives_variance = np.sum(log_derivatives_variance_f, axis=0) / number_of_samples

        F_mean = np.matmul(np.transpose(log_derivatives_mean), log_derivatives_mean) / number_of_samples
        F_variance = np.matmul(np.transpose(log_derivatives_variance), log_derivatives_variance) / number_of_samples

        mean = mean - learning_rate * (np.matmul(np.linalg.inv(F_mean),j_derivatives_mean))
        variance = variance - learning_rate * (np.matmul(np.linalg.inv(F_variance),j_derivatives_variance))
        variance += 0.01

        #For plots
        idx = np.argsort(evaluated_samples)
        best_fitness_samples[i] = samples[idx[0]]
        worst_fitness_samples[i] = samples[idx[-1]]
        #Plot in 2 dimensions
        # plot_samples_on_contour(samples, objective_function)
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
    #prints for the report:
    print("Avg best fitness:" + str(np.min(best_fitness/number_of_runs)))
    print("Avg worst fitness:" + str(np.min(worst_fitness/number_of_runs)))
    #CHECK: is the order of the operation (sum, mean, sphere_test) correct?
    plt.plot(np.arange(0,number_of_generations), best_fitness / number_of_runs, "r", label="best fitness")
    plt.plot(np.arange(0,number_of_generations), worst_fitness/number_of_runs, "b", label="worse fitness")
    plt.title('Number of Generations: ' + str(number_of_generations) + '\nPopulation number: ' + str(number_of_samples) + '\nNumber of features: ' + str(number_of_features)\
        +'\n Elite set: '+ str(elite_set))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(loc="upper right")
    plt.show()

def cmaes(mean, covariance_matrix, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function):
    i = 0
    best_fitness_samples = np.zeros((number_of_generations, number_of_features))
    worst_fitness_samples =  np.zeros((number_of_generations, number_of_features))
    while(i < number_of_generations):
        samples = np.random.multivariate_normal(mean, covariance_matrix, (number_of_samples))
        # Remember without minus is minimizing
        idx = np.argsort(objective_function(samples))
        # Update covariance first avoids this backup
        covariance_matrix = np.zeros((number_of_features, number_of_features))
        for ii in range(number_of_features):
            for j in range(number_of_features):
                for sample in samples[idx][:elite_set]:
                    covariance_matrix[ii,j] += (sample[ii] - mean[ii])*(sample[j] - mean[j])
        covariance_matrix = covariance_matrix / elite_set
        mean = np.mean(samples[idx][:elite_set], axis=0)
        #For plots
        best_fitness_samples[i] = samples[idx[0]]
        worst_fitness_samples[i] = samples[idx[-1]]

        #Plot in 2 dimensions
        # plot_samples_on_contour(samples, objective_function)
        
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
    #prints for the report:
    print("Avg best fitness:" + str(np.min(best_fitness/number_of_runs)))
    print("Avg worst fitness:" + str(np.min(worst_fitness/number_of_runs)))
    #CHECK: is the order of the operation (sum, mean, sphere_test) correct?
    plt.plot(np.arange(0,number_of_generations), best_fitness / number_of_runs, "r", label="best fitness")
    plt.plot(np.arange(0,number_of_generations), worst_fitness/number_of_runs, "b", label="worse fitness")
    plt.title('Number of Generations: ' + str(number_of_generations) + '\nPopulation number: ' + str(number_of_samples) + '\nNumber of features: ' + str(number_of_features)\
        +'\n Elite set: '+ str(elite_set))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(loc="upper right")
    plt.show()

        
        






### Sphere test plot, change with np.random.normal()
# x = np.linspace(-5,5,100)
# y = np.linspace(-5,5,100)
#visualize_sphere(x,y)
#g = np.linspace(-5,5,100)

### cem test 
number_of_generations = 50
number_of_samples = 100
number_of_features = 20
elite_set = 20
objective_function = sphere_test
number_of_runs = 3
mean = np.random.uniform(-5,5,number_of_features)
variance = np.random.uniform(0,5,number_of_features)
learning_rate = 0.001

#covariance matrix. CHECK: is correct to use normal?
covariance_matrix = np.diag(np.random.uniform(-5,5,number_of_features))


# cem(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function)
# cem_d(mean, variance, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function, number_of_runs)
# nes_d(mean, variance, number_of_generations, number_of_samples, number_of_features, objective_function, learning_rate, number_of_runs)
cmaes_d(mean, covariance_matrix, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function, number_of_runs)
#cmaes(mean, covariance_matrix, number_of_generations, number_of_samples, number_of_features, elite_set, objective_function)
# nes(mean, variance, number_of_generations, number_of_samples, number_of_features, objective_function, learning_rate)
# visualize_obj_function(objective_function)

# plt.show()

## Exercise 1
# visualize_obj_function(objective_function)
# samples = np.random.uniform(-5,5,(100,2))
# plt.scatter(samples[:, 0], samples[:, 1], c= objective_function(samples), s=500, cmap='RdGy')
# plt.colorbar()
# plt.show()
##


