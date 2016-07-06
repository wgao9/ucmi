import numpy.random as nr
import ucmi 

def demo1():
	print "Demo 1 : Joint Gaussian"
	x = nr.normal(0,1,[1000,1])
	y = nr.normal(0,1,[1000,1])
	print "I(X;Y) = ", ucmi.mi(x,y)
	print "UMI(X;Y) = ", ucmi.umi(x,y)
	print "CMI(X;Y) = ", ucmi.cmi(x,y)

def demo2():
	print "Demo2 : Beta + Gaussian Noise"
	x = nr.beta(1.5,1.5,[1000,1])
	y = x + nr.normal(0,0.3,[1000,1])
	print "I(X;Y) = ", ucmi.mi(x,y)
	print "UMI(X;Y) = ", ucmi.umi(x,y)
	print "CMI(X;Y) = ", ucmi.cmi(x,y)

if __name__ == '__main__':
	demo1()
	demo2()


