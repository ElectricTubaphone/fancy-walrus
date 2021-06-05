package simpleNetwork;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class simpleNetwork {
	private Layer[] layers;
	private int layerCount;
	int activation;
	float learningRate = 0.05f;
	
	
	
	
	/*
	 * Constructors for simpleNetwork
	 */
	public simpleNetwork(IntBuffer header, FloatBuffer body) {
		//store number of layers and initialize array
		layerCount = header.get();//get number of layers
		layers = new Layer[layerCount];

		//store activation function id 
		this.activation = header.get();
		
		//retrieve each layer's node count
		int[] layernodecount = new int[layerCount];
		for(int i = 0; i < layerCount; ++i) {
			layernodecount[i] = header.get();
		}
		
		//individually initialize each element of the array
		for(int i = 0; i < layerCount; ++i) {
			if(i + 1 < layernodecount.length) {
			layers[i] = new Layer(layernodecount[i], layernodecount[i+1], activation);
			} else {
				layers[i] = new Layer(layernodecount[i], 0, activation);
			}
			layers[i].randomize_nodes(-.8f, .8f, -0.3f, 0.3f, body.limit());
		}
		
		//load weights and biases for the network
		load_floatbuffer(body);
	}
	
	public simpleNetwork(int[] layernodecount, Activation.ACTIVATION activationFunction) {
		//store number of layers and initialize array
		layerCount = layernodecount.length;
		layers = new Layer[layerCount];
		
		int activation = 0;
		switch(activationFunction) {
		case SIGMOID:
			activation = 0;
			break;
		case TANH:
			activation = 1;
			break;
		case RELU:
			activation = 2;
			break;
		case LEAKY_RELU:
			activation = 3;
			break;
		}
		
		//individually initialize each element of the array
		for(int i = 0; i < layerCount; ++i) {
			if(i + 1 < layernodecount.length) {
				layers[i] = new Layer(layernodecount[i], layernodecount[i+1], activation);
			} else {
				layers[i] = new Layer(layernodecount[i], 0, activation);
			}
			layers[i].randomize_nodes(-.8f, .8f, -0.3f, 0.3f, layernodecount[layernodecount.length-1] * layernodecount[layernodecount.length-2]);
		}
	}
	
	public simpleNetwork(int[] layernodecount, Activation.ACTIVATION activationFunction, float wmin, float wmax, float bmin, float bmax, float learningrate) {
		//store number of layers and initialize array
		layerCount = layernodecount.length;
		layers = new Layer[layerCount];
		learningRate = learningrate;
		
		int activation = 0;
		switch(activationFunction) {
		case SIGMOID:
			activation = 0;
			break;
		case TANH:
			activation = 1;
			break;
		case RELU:
			activation = 2;
			break;
		case LEAKY_RELU:
			activation = 3;
			break;
		}
		
		//individually initialize each element of the array
		for(int i = 0; i < layerCount; ++i) {
			if(i + 1 < layernodecount.length) {
				layers[i] = new Layer(layernodecount[i], layernodecount[i+1], activation);
			} else {
				layers[i] = new Layer(layernodecount[i], 0, activation);
			}
			layers[i].randomize_nodes(wmin, wmax, bmin, bmax, layernodecount[layernodecount.length-1] * layernodecount[layernodecount.length-2]);
		}
	}
	
	public simpleNetwork(int layercount, int nodecount, Activation.ACTIVATION activationFunction) {
		//store number of layers and initialize array
		layerCount = layercount;
		layers = new Layer[layercount];
		
		int activation = 0;
		switch(activationFunction) {
		case SIGMOID:
			activation = 0;
			break;
		case TANH:
			activation = 1;
			break;
		case RELU:
			activation = 2;
			break;
		case LEAKY_RELU:
			activation = 3;
			break;
		}
		
		//individually initialize each element of the array
		for(int i = 0; i < layercount; ++i) {
			layers[i] = new Layer(nodecount, nodecount, activation);
			layers[i].randomize_nodes(-.8f, .8f, -0.3f, 0.3f, layercount*nodecount);
		}
	}
	
	
	
	
	/*
	 * Data format operations for transferring network data (Saving and Loading)
	 */
	
	public IntBuffer unload_header() {
		int size = 0;
		size += 1;//layer count
		size += 1;//activation function
		size += layers.length;//Array(LayerNodeCount)
		IntBuffer buf = IntBuffer.allocate(size);
		buf.put(layers.length);//store number of layers at head of header
		buf.put(activation);//store activation function id
		for(int i = 0; i < layers.length; ++i) {//then store node size of each layer
			buf.put(layers[i].getNodeCount());
		}
		buf.flip();//prepare buffer for reading
		return buf;
	}
	
	public FloatBuffer unload() {
		/*
		 * Unloading Format:
		 * LayerCount, Activation function, Array(LayerNodeCount), 
		 * THIS METHOD: Layers.Array(Nodes.Array(Weight.Array(), bias))
		 */
		FloatBuffer buf = FloatBuffer.allocate(unload_size());
		for(int i = 0; i < layers.length-1; ++i) {
			for(int j = 0; j < layers[i].getNodeCount(); ++j) {
				for(int k = 0; k < layers[i+1].getNodeCount(); ++k) {
					buf.put(layers[i].getNextWeight(j, k));
				}
				buf.put(layers[i].getBias(j));
			}
		}
		for(int i = 0; i < layers[layers.length-1].getNodeCount(); ++i) {
			buf.put(layers[layers.length-1].getBias(i));
		}
		buf.flip();//prepare buffer to be read
		return buf;
	}
	
	private void load_floatbuffer(FloatBuffer buf) {
		//expect buf to be already flipped
		for(int i = 0; i < layers.length-1; ++i) {
			for(int j = 0; j < layers[i].getNodeCount(); ++j) {
				for(int k = 0; k < layers[i+1].getNodeCount(); ++k) {
					layers[i].setWeight(j, k, buf.get());
				}
				layers[i].setBias(j, buf.get());
			}
		}
		for(int i = 0; i < layers[layers.length-1].getNodeCount(); ++i) {
			layers[layers.length-1].setBias(i, buf.get());
		}
	}
	
	private int unload_size() {
		int size = 0;
		for(int i = 0; i < layers.length-1; ++i) {
			size += layers[i].getNodeCount() * (layers[i+1].getNodeCount() + 1);//Nodes.Array(Weight.Array(), bias)
		}
		size += layers[layers.length-1].getNodeCount();//biases for final layer
		return size;
	}
	
	
	
	
	
	//Debug output
	public void Debug_Print() {
		System.out.println(" === Network ===");
		for(int i=0;i<layerCount;++i) {
			System.out.print("L" + i + ": ");
			for(int j = 0; j < layers[i].getNodeCount(); ++j) {
				layers[i].Debug_Print(j);
				System.out.print("\n");
			}
		}
	}
	
	
	
	
	
	
	
	
	/*
	 * Propagation Methods
	 */
	
	//propagate using input array.
	public void propagate(float[] input) {
		setInputValues(input);
		propagate();
	}
	
	//Propagate based on stored input values. Can set input values using setInputValues, or by using the propagate(float[] input) function instead.
	public void propagate() {
		//this version expects input values to be set via relevant function
		//loop through each layer
		for(int i = 0; i < layerCount-1; ++i) {
			layers[i].propagate(layers[i+1]);
		}
		
	}
	
	//propagation with results as output
	public float[] run(float[] input) {
		propagate(input);
		return layers[layerCount-1].getNodeValues();
	}

	
	
	
	
	
	
	
	
	
	
	/*
	 * Backpropagation and Training Methods.
	 */
	
	
	//train using a single example
	public void train(float[] input, float[] target) {
		propagate(input);
		backpropagateTarget(target);
	}
	
	
	//run a batch of training examples
	public void train(float[][] input, float[][] target) {
		for(int i = 0; i < input.length && i < target.length && i < 200; ++i) {
			propagate(input[i]);
			backpropagateTarget(target[i]);
		}
	}
	
	
	
	
	//Standard backpropagate using targets
	public void backpropagateTarget(float target[]) {
		backpropagateTarget(target, learningRate);
	}
	
	
	
	
	//Backpropagate based on difference between predicted values and outputs
	public void backpropagateTarget(float target[], float learningRate) {
		layers[layerCount-1].setTargetError(target);
		for(int i = layerCount - 1; i > 0; --i) 
			layers[i].backpropagateFormerNode(layers[i-1], learningRate);
	}
	
	
	
	
	//Backpropagate without implementing weight and bias changes (Used for determining the error of the first layer)
	public void backpropagateTargetChangeless(float target[]) {
		layers[layerCount-1].setTargetError(target);
		for(int i = layerCount - 1; i > 0; --i) 
			layers[i].backpropagateFormerNode_NoChange(layers[i-1]);
	}
	
	
	
	
	//Backpropagate based on error value determined externally
	public void backpropagateError(float[] error) {
		layers[layerCount-1].setNodeError(error);
		for(int i = layerCount - 1; i > 0; --i) 
			layers[i].backpropagateFormerNode(layers[i-1]);
	}
	
	
	
	
	//Backpropagate using the error of the first layer
	public void backpropagateLoop() {
		loopActivationCost();
		for(int i = layerCount - 1; i > 0; --i) 
			layers[i].backpropagateFormerNode(layers[i-1]);
	}
	
	
	
	
	//Assign error of first layer to last layer (for special instances)
	public void loopActivationCost() {
		for(int i = 0; i < layers[layers.length - 1].getNodeValues().length; ++i) {
			layers[layers.length-1].setNodeActivationCost(i, layers[0].getNodeActivationCost(i));
		}
	}
	

	
	
	
	
	
	/*
	 * Set variables for propagation and training
	 */
	
	public void setLearningRate(float rate) {
		learningRate = rate;
	}
	
	public void setInputValues(float[] values) {
		layers[0].setNodeValues(values);
	}
	
	
	
	
	
	
	/*
	 * Get Input and Output Data
	 */
	
	public float[] getInputError() {
		return layers[0].getNodeError();
	}
	
	public float[] getInputValues() {
		return layers[0].getNodeValues();
	}
	
	public float[] getOutputValues() {
		return layers[layerCount-1].getNodeValues();
	}
	
}
