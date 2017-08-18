from scipy import signal as sg

"""
image_shape=(batch size,W,H,Number of Channels)
filter_shape=(Number of filters,W,H,Number of Channels)
"""
class conv(object):
    def __init__(self,image_shape,filter_shape):
        self.filter_shape=filter_shape
        self.input_shape=image_shape
        
        self.weights = np.random.normal(size=filter_shape)
        self.biases = np.full((filter_shape[0],1),0.1)
        
        self.params=[self.weights, self.biases]
        
    def convolve(self, minibatch,flag=1):
        
        size_out=(self.input_shape[1]-self.filter_shape[1]) + 1
        self.unpooled_output=np.zeros((self.input_shape[0],self.filter_shape[0],size_out,size_out))
        
        """
        self.convolution(n=self.input_shape[0],f=self.filter_shape[1],s=self.stride,d=self.filter_shape[0],
                    inpt=input_image)
        """
        if flag==1:
            self.convolution_implmeneted_1(inpt=minibatch,type_="valid")
        else :
            self.convolution_implmeneted_2(inpt=minibatch,type_="valid")
        self.pooling()
   

    def convolution_implmeneted_1(self,inpt, type_ ):
        for j in xrange(self.input_shape[0]):
            for i in xrange(self.filter_shape[0]):
                self.unpooled_output[j,i]=sg.convolve(inpt[j][0],self.weights[i],type_).reshape(self.unpooled_output.shape[2:])
                + self.biases[i];
    def convolution_implmeneted_2(self,inpt, type_ ):
        for j in xrange(self.input_shape[0]):
            for i in xrange(self.filter_shape[0]):
                self.unpooled_output[j,i]=sg.convolve(inpt[j],self.weights[i],type_).reshape(self.unpooled_output.shape[2:])
                + self.biases[i];
    
    
    
    """    
        def convolution(self,n,f,s,d,inpt):
            for i in xrange(d):
                for x in xrange (0,n-f+1,s):
                    for y in xrange (0,n-f+1,s):
                        roi=inpt[x:x+f,y:y+f]
                        self.unpooled_output[x,y,i]=relu((roi*self.weights[i]).sum()+self.biases[i])

    """    

    def pooling(self):
        n=self.unpooled_output.shape[2]
        f=2
        s=2
        d=self.unpooled_output.shape[1]
        self.map=np.zeros(self.unpooled_output.shape)
        size_out=(n-f)/s+1
        self.output=np.zeros((self.input_shape[0],d,size_out,size_out))
        self.pooled_shape=(self.input_shape[0],size_out,size_out,d)
        for j in xrange(self.input_shape[0]):
            for i in xrange(d):
                ix=0
                for x in xrange (0,n-f+1,s):
                    iy=0
                    for y in xrange (0,n-f+1,s):
                        roi=self.unpooled_output[j,i,x:x+f,y:y+f]
                        self.map[j,i,x:x+f,y:y+f]=np.argmax()
                        self.output[j,i,ix,iy]=roi.max()
                        iy+=1
                    ix+=1
        self.output=self.output.reshape(self.pooled_shape)
def relu(z):
    return np.maximum(z,0)