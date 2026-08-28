import numpy as np
'''
a=np.array([1,2,3],dtype='int16')
print(a);

b=np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)

c=np.array([3,4,5,6,7,8,9])

print(a.ndim) #getting dimension of arr
print(b.shape) #getting no. of rows and cols
print(b.dtype) #get type
print(a.itemsize) #get size - number of bytes used by one element
print(b.size) #number of elements

#a.size*a.itemsize=a.nbytes
print(b[0,2]) #equals to b[0,-1]; negative notation
print(b[1,:]) #get specific row; reverse for col
print(c[2:5:2 ]) #for 3-D array, use c[1,startindex:stopindex:stepval]

#changing element is possible
d=np.zeros(5) #any dimension possible; similarly, np.ones
print(d)
#to fill woth any other number:np.full((dimension),num)->viz. np.full(2,99)
np.full(b.shape,9) #equals np.full_like(b,9) ->it basically takes shape of a specific array

#rand deci for 0<num<1; ->put dimension
e=np.random.rand(3,5) #we also use np.random.random_sample(2,3) ->both r same
print(e)
#rand integer
f=np.random.randint(7, size=(2,5)) #generate rand from 0-7(exclude 7)
print(f)

g=np.identity(4)
print(g)
h=np.array([1,2,3])
g=np.repeat(h,3,axis=0)
print(g)

i=np.ones((5,5))
j=np.zeros((3,3))
j[1,1]=9
i[1:4,1:4]=j #4 is exclusive
print(i)

k=np.array([1,2,3])
l=k.copy() #l=k doesn't work since it changes the og array thereby printing 6,2,3 for both arrays
l[0]=6
print(k)
print(l)

#element-wise computation
m=np.array([4,5,6,7])
print(m+3) #same as m+=2
print(m-3)
print(m/3)
print(m*3)
print(m**2)
n=([1,2,3,4])
print(m+n)
print(np.sin(n))

o=np.ones((3,3))
p=np.full((3,2),2)
#print(np.matmul(o,p)) #matrix multiplication
print(np.linalg.det(o)) #determinant
n=([1,2,3,4])
print(np.min(n)) #similarly, max; for nD array, we can do like min(a,axis=3)
print(np.sum(n))
b=np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
q=b.reshape((1,6))
print(q)

o=np.ones((3))
p=np.full((3,2))
print(np.vstack(o,p,p))

v1=np.array([1,2,3,4])
v2=np.array([5,6,7,8])
print(np.vstack([v1,v2,v2,v1]))

h1=np.ones((2,4))
h2=np.zeros((2,2))
print(np.hstack((h1,h2)))

filedata=np.genfromtxt('data.txt',delimiter=',') #file e numbers r split using ',' so delimiter=','
print(filedata)
#it prints them as float despite being in integer
#to print as integer, we do: filedata=filedata.astype('int32')

#boolean masking and advanced indexing
print((filedata>50) & (filedata<100))
print(filedata[filedata>50]) #similarly, np.any(filedata>50,axis=0)=np.all(filedata>50,axis=0)
'''
