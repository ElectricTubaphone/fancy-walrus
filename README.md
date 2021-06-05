# fancy-walrus
A simple fully connected neural network in Java, paired with a classifier class. The classes can easily be integrated into other projects.
## An example of the network

Here is a demonstration of a simple implementation of the class. In it, a network is trained to model the sine function for input values between 0 and 1.

```Java
//create the layer array to specify the size of each layer
int[] layer_array = { 1, 6, 6, 2 };

//initialize the network using layer_array and using the sigmoid function as the activation.
simpleNetwork network = new simpleNetwork(layer_array, Activation.ACTIVATION.SIGMOID);


public static void main(String Args[]) {
  Main run = new Main();
  
  //train 10,000 propagations of the network
  for(int i = 0; i < 10000; ++i) {
    
        //using a random input
        float[] input = { ((float) Math.random()) };
			
        //and training the output to map the input to sine
        float[] output = { (float)Math.sin((double)input[0])} ;
      
        //run training using the input, and training against the desired output
        run.network.train(input, output);
    }
    
    //perform the same input and target set-up.
    float[] input = { ((float) Math.random()) };
    float[] target = { (float)Math.sin((double)input[0])} ;
    
    //get the actual output of the network
    float outputValue = run.network.run(input)[0];
    
    //report the actual output of the network, and compare against the target value.
    System.out.println("For input " + input[0] + ", the network mapped to " + outputValue + ". The correct value is " + target[0] + "." );
}
```
This is one program using the neural net. As seen in the above code, input into and output from the network both take the form of float arrays. As long as you can create float arrays which contain target values, and can keep track of corresponding input arrays, then you can easily train the network. Ideally, all input and target outputs are normalized to exist in a range similar to -1 to +1.

## simpleNetwork
The simpleNetwork class is a fully connected neural network. The class has several constructors, depending on which parameters you're concerned with tweaking.

There is 
```Java
simpleNetwork(int[] layernodecount, Activation.ACTIVATION activationFunction)
```
Which creates a network in which each layer has the number of nodes specified in its respective index in layernodecount.
ActivationFunction can be either ACTIVATION.SIGMOID, ACTIVATION.TANH, ACTIVATION.RELU, and ACTIVATION.LEAKY_RELU.

Then, there's
```Java
simpleNetwork(int layercount, int nodecount, Activation.ACTIVATION activationFunction)
```
This is similar to the previous network, except the number of layers is equal to layercount, and every layer has the same number of nodes specified by nodecount.

Another constructor is the following...
```Java
simpleNetwork(int[] layernodecount, Activation.ACTIVATION activationFunction, float wmin, float wmax, float bmin, float bmax, float learningrate)
```
Where wmin and wmax are the minimum and maximum initial values of the weights in the network, and bmin and bmax the minimum and maximum values for the biases. Furthermore, there is the learningRate, which should be some number below 1 (such as 0.05).

### Public Methods in simpleNetwork
While not a comprehensive list, these methods should comprise the entirety of the useful functions of the simpleNetwork:

```Java
public void propagate(float[] input)
```
This method takes an input float array, which should be as long as the number of nodes in the first layer. Nothing is returned, but the output can be accessed using simpleNetwork.getOutputValues(). 

```Java
public void setInputValues(float[] values)
```
This method takes the inputted values float array and stores it to be used in the next propagation, as long as no other input is introduced in the mean time.

```Java
public void propagate()
```
This method propagates based on whatever input has been fed into the class last. You can feed an input into the class with simpleNetwork.setInputValues(float[] values). 

```Java
public float[] run(float[] input)
```
This method takes the input value array and returns an array of the output values. This is likely the most valuable propagation method.

```Java
public void setLearningRate(float rate)
```
This resets the rate to whatever is specified, so if for whatever reason during the course of your program you need the descent towards an optimum to go faster and with less accuracy, or need to become more precise as you approach the optimum, this method should accomodate that need.

```Java
public float[] getInputError()
```
This returns a float array, which is comprised of the error of each input node calculated in backpropagation through the train methods.

```Java
public float[] getInputValues()
```
Returns the most recent input float array entered into the class.

```Java
public void train(float[] input, float[] target)
```
Runs one propagation using the input array, and then performs one round of backpropagation using the target array.

```Java
public void train(float[][] input, float[][] target)
```
Computes every input and backpropagates for each target in the provided arrays. If one array is bigger than the other, then only indices which have both an input and target value will be computed.

```Java
public void backpropagateTarget(float target[])
```
Backpropagates and applies weight and bias changes based on the most recent propagation.

```Java
public void backpropagateError(float[] error)
```
Backpropagates on the basis of externally specified error values. The array of error values should be equal in length to the number of nodes in the last layer.


```Java
public void Debug_Print()
```
Debug_Print prints the values of the each node, weight, and bias. It is not especially useful except in the development of these classes.

# Saving and Loading to/from file
```Java
public static void simpleNetworkFileUtility.saveFile(simpleNetwork network, String path)
```
This method takes a network and a path and then saves the network data to said file.

```Java
public static simpleNetwork simpleNetworkFileUtility.loadFile(String path) throws IOException, FileNotFoundException
```
This method reads a file and returns a simpleNetwork which can be assigned to a simpleNetwork object.
