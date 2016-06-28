import numpy.random as nr
import ucmi 

def main():
	x = nr.normal(0,1,[1000,1])
	y = nr.normal(0,1,[1000,1])
	print "I(X;Y) = ", ucmi.mi(x,y)
	print "UMI(X;Y) = ", ucmi.umi(x,y)
	print "CMI(X;Y) = ", ucmi.cmi(x,y)

if __name__ == '__main__':
	main()


