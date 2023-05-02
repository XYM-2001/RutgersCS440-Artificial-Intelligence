import random
def select_sample(images, labels, sample_size):
    combined = list(zip(images, labels))
    sample = random.sample(combined, sample_size)
    sampleimage, samplelabel = zip(*sample)
    return sampleimage, samplelabel

a = [1,2,3]
b = [4,5,6]
ar,br = select_sample(a,b, 2)
print(type(ar))