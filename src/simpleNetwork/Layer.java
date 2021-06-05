package simpleNetwork;


import datasetUtils.RandomGenerator;

class Layer {
	private Node[] nodes;
	int nodeCount;
	int activationFunction;
	
	
	/*
	 * Constructor
	 */
	
	public Layer(int nodecount, int nextnodecount, int activation_function) {
		//initialize nodes array and store size of array
		nodes = new Node[nodecount];
		nodeCount = nodecount;
		//initialize individual nodes
		for(int i = 0; i < nodeCount; ++i)
			nodes[i] = new Node(nextnodecount);
		//set the activation function
		//0 = sigmoid; 1 = tanh; 2 = ReLU; 3 = Leaky ReLU
		this.activationFunction = activation_function;
		//if value is out of bounds, set pointing to sigmoid
		if(activationFunction < 0 || activationFunction > 3) {
			activationFunction = 0;
		}
		//Initialize the values inside of the nodes
		initialize_nodes();
	}
	
	
	
	/*
	 *  Miscellaneous Initialization Methods
	 */

	private void initialize_nodes(){
		randomize_nodes(-2.f, 3.f, -0.8f, 0.8f, 18);
	}
	
	
	public void setTargetError(float[] target) {
		//iterate through each node, and set ActivationCost equal to (actual-target)
		for(int i = 0; i < target.length && i < nodeCount; ++i) {
			nodes[i].setActivationCost(nodes[i].getValue() - target[i]);
		}
	}
	
	
	
	
	
	
	
	
	
	
	
	
	/*
	 * Propagation
	 */
	
	public void propagate(Layer nextLayer) {
		nextLayer.propagate_from_former_layer(this);
	}
	
	public void propagate_from_former_layer(Layer formerLayer) {
		//iterate through open nodes in this array
		for(int i = 0; i < nodeCount; ++i) {
			//make temporary value to store progressive value, set to bias
			float value = getBias(i);
			//iterate through all former nodes
			for(int j = 0; j < formerLayer.getNodeCount(); ++j)
			{
				//take each value from node and multiply it by the weight to current highest node
				value += formerLayer.getNextWeight(j, i) * formerLayer.getNodeValue(j);
			}
			
			//store the input of the activation for later backpropagation
			setNodeInput(i, value);
			
			//activate the value ( run the activation function )
			switch ( activationFunction ) {
			case 0:
				value = Activation.sigmoid(value);
				break;
			case 1:
				value = Activation.tanh(value);
				break;
			case 2:
				value = Activation.ReLU(value);
				break;
			case 3:
				value = Activation.leaky_ReLU(value);
				break;
			};
			//now set node value to value calculated
			setNodeValue(i, value);
		}
	}
	
	
	
	
	
	
	
	
	
	
	/*
	 * Backpropagation Methods
	 */
	
	
	
	
	public void backpropagate(Layer formerLayer) {
		backpropagateFormerNode(formerLayer);
	}
	
	public void backpropagate_node(int node, Layer formerLayer, float target) {
		int formerNodeCount = formerLayer.getNodeCount();
		//get dE_total/dOut_o2
		float dETotal_dOut = nodes[node].getValue() - target;
		float dOut_dIn = Activation.sigmoid_derivative(getNodeInput(node));
		switch(activationFunction) {
		case 0:
			//already did the sigmoid derivative, break
			break;
		case 1:
			dOut_dIn = Activation.tanh_derivative(getNodeInput(node));
			break;
		case 2:
			dOut_dIn = Activation.ReLU_derivative(getNodeInput(node));
			break;
		case 3:
			dOut_dIn = Activation.leaky_ReLU_derivative(getNodeInput(node));
			break;
		}
		
		//iterate through nodes in former layer and store the respective weight changes
		for(int i = 0; i < formerNodeCount; ++i) {
			formerLayer.setWeightChange(i, node, 
					dETotal_dOut*dOut_dIn*formerLayer.getNodeValue(i));
		}
		//TODO: setBiasChange implementation
	}
	
	
	public void backpropagateFormerNode(Layer formerLayer) {
		backpropagateFormerNode(formerLayer, 0.05f);
	}
	
	public void backpropagateFormerNode(Layer formerLayer, float learningRate) {
		//dETotal_dwback
		// dEtotal/dOutback = dEo1/dOutback + dEo2/dOutback + ...
		// dEo2/dOutback = dEo2/dIno2 * dIno2/dOutback
		// dEo2/dIno2 = dEo2/dOuto2 * dOuto2/dIno2
		// dEo2/dOuto2 = actual-target
		// dOuto2/dIno2 = sigmoid'(Outo2)
		// dIno2/dOutback = connecting weight value
		
		//dEtotal/dw4 = dETotal/dOutback * dOutback/dInback * dInback/dw4
		
		//dIn//dw4 = input value of receiving node
		
		//calculate activation cost derivative for each node in former layer, based on present layer
		//running this method assumes that the present layer already calculated the activation cost derivative
		
		
		for(int i = 0; i < formerLayer.getNodeCount(); ++i) {
			formerLayer.setNodeActivationCost(i, derivate_activation(i, formerLayer));
		}
		
		switch(activationFunction) {
		case 1:
			for(int i = 0; i < formerLayer.getNodeCount(); ++i) {
				for(int j = 0; j < nodeCount; ++j) {
					formerLayer.setWeightChange(i, j, formerLayer.getNodeValue(i) * Activation.tanh_derivative(nodes[j].getInput()) * nodes[j].getActivationCost() );
				}
			}
			for(int i = 0; i < nodeCount; ++i) {
				nodes[i].setBiasChange(Activation.tanh_derivative(nodes[i].getInput()) * nodes[i].getActivationCost());
			}
			break;
		case 2:
			for(int i = 0; i < formerLayer.getNodeCount(); ++i) {
				for(int j = 0; j < nodeCount; ++j) {
					formerLayer.setWeightChange(i, j, formerLayer.getNodeValue(i) * Activation.ReLU_derivative(nodes[j].getInput()) * nodes[j].getActivationCost() );
				}
			}
			for(int i = 0; i < nodeCount; ++i) {
				nodes[i].setBiasChange(Activation.ReLU_derivative(nodes[i].getInput()) * nodes[i].getActivationCost());
			}
			break;
		case 3:
			for(int i = 0; i < formerLayer.getNodeCount(); ++i) {
				for(int j = 0; j < nodeCount; ++j) {
					formerLayer.setWeightChange(i, j, formerLayer.getNodeValue(i) * Activation.leaky_ReLU_derivative(nodes[j].getInput()) * nodes[j].getActivationCost() );
				}
			}
			for(int i = 0; i < nodeCount; ++i) {
				nodes[i].setBiasChange(Activation.leaky_ReLU_derivative(nodes[i].getInput()) * nodes[i].getActivationCost());
			}
			break;
		case 0:
		default:
			for(int i = 0; i < formerLayer.getNodeCount(); ++i) {
				for(int j = 0; j < nodeCount; ++j) {
					formerLayer.setWeightChange(i, j, formerLayer.getNodeValue(i) * Activation.sigmoid_derivative(nodes[j].getInput()) * nodes[j].getActivationCost() );
				}
			}
			for(int i = 0; i < nodeCount; ++i) {
				nodes[i].setBiasChange(Activation.sigmoid_derivative(nodes[i].getInput()) * nodes[i].getActivationCost());
			}
			break;
		}
		//now that each weight has been assigned its change, increment the weight change index to next value for storing more backpropagation data
		formerLayer.applyWeightChangeAll(learningRate);
		applyBiasChangeAll(learningRate);
		
		
	}
	
	public void backpropagateFormerNode_NoChange(Layer formerLayer) {
		
		//calculate activation cost derivative for each node in former layer, based on present layer
		//running this method assumes that the present layer already calculated the activation cost derivative
		
		
		for(int i = 0; i < formerLayer.getNodeCount(); ++i) {
			formerLayer.setNodeActivationCost(i, derivate_activation(i, formerLayer));
		}
		//Don't make changes to weights. This is merely a means to calculate the error for each output of the generator network
		
		
	}
	
	
	
	
	
	
	
	
	/*
	 * Gradient calculation methods
	 */
	
	
	
	
	private float derivate_activation(int formerNode, Layer formerLayer) {
		float cost = 0.f;
		for(int i = 0; i < nodeCount; ++i) {
			float dCo = 0.f;
			float z = nodes[i].getInput();
			dCo = formerLayer.getNextWeight(formerNode, i);
			switch(activationFunction) {
			case 1:
				dCo *= Activation.tanh_derivative(z);
				break;
			case 2:
				dCo *= Activation.ReLU_derivative(z);
				break;
			case 3:
				dCo *= Activation.leaky_ReLU_derivative(z);
				break;
			case 0:
			default:
				dCo *= Activation.sigmoid_derivative(z);
			}
			dCo *= nodes[i].getActivationCost();
			cost += dCo;
		}
		return cost;
	}
	
	public float getOuterWeightChange(int node, int node2, float target, Layer formerLayer) {
		float dETotal_dOut = nodes[node2].getValue() - target;
		float dOut_dIn = Activation.sigmoid_derivative(getNodeInput(node2));
		switch(activationFunction) {
		case 0:
			//already did the sigmoid derivative, break
			break;
		case 1:
			dOut_dIn = Activation.tanh_derivative(getNodeInput(node2));
			break;
		case 2:
			dOut_dIn = Activation.ReLU_derivative(getNodeInput(node2));
			break;
		case 3:
			dOut_dIn = Activation.leaky_ReLU_derivative(getNodeInput(node2));
			break;
		}
		float dIn_dWOut = formerLayer.getNodeValue(node);
		return dETotal_dOut * dOut_dIn * dIn_dWOut;
	}
	
	public float getOuterBiasChange(int node, int node2, float target, Layer formerLayer) {
		float dETotal_dOut = nodes[node2].getValue() - target;
		float dOut_dIn = Activation.sigmoid_derivative(getNodeInput(node2));
		switch(activationFunction) {
		case 0:
			//already did the sigmoid derivative, break
			break;
		case 1:
			dOut_dIn = Activation.tanh_derivative(getNodeInput(node2));
			break;
		case 2:
			dOut_dIn = Activation.ReLU_derivative(getNodeInput(node2));
			break;
		case 3:
			dOut_dIn = Activation.leaky_ReLU_derivative(getNodeInput(node2));
			break;
		}
		float dIn_dWOut = 1.f;
		return dETotal_dOut * dOut_dIn * dIn_dWOut;
	}
	
	
	
	
	
	
	
	
	
	
	
	/*
	 * Weight and Bias Methods
	 */
	
	
	
	public void applyBiasChangeAll(float learningrate) {
		for (int i = 0 ; i < nodeCount; ++i) {
			applyBiasChange(i, learningrate);
		}
	}
	
	public void applyBiasChange(int node, float learningrate) {
		nodes[node].applyBiasChange(learningrate);
	}
	
	public void applyWeightChangeAll(float learningrate) {
		for(int i = 0; i < nodeCount; ++i)
			applyWeightChange(i, learningrate);
	}
	
	public void applyWeightChange(int node, float learningrate) {
			nodes[node].applyWeightChange(learningrate);
	}
	
	public void setWeightChange(int node, int weight, float change) {
		nodes[node].setWeightChange(weight, change);
	}
	
	public void setBiasChange(int node, float change) {
		nodes[node].setBiasChange(change);
	}
	

	public void setWeight(int node, int nextnode, float weight) {
		nodes[node].setWeight(nextnode, weight);
	}
	
	public void setBias(int node, float bias) {
		nodes[node].setBias(bias);
	}
	
	public float getBias(int node) {
		return nodes[node].getBias();
	}
	
	public float getNextWeight(int node, int nextnode) {
		return nodes[node].getWeight(nextnode);
	}
	
	
	
	
	
	
	
	
	
	/*
	 * Access node information, such as values, inputs, and count
	 */
	
	
	
	//Node Input
	
	
	private void setNodeInput(int node, float input) {
		nodes[node].setInput(input);
	}
	
	
	
	
	private float getNodeInput(int node) {
		return nodes[node].getInput();
	}
	
	
	
	//Node Count
	
	
	public int getNodeCount() {
		return nodeCount;
	}

	
	
	
	//Node Values
	
	public float getNodeValue(int node) {
		if(node < nodeCount) {
			return nodes[node].getValue();
		}
		else
		{
			return -1.f;
		}
	}

	
	public void setNodeValue(int node, float value) {
		nodes[node].setValue(value);
	}
	
	
	public float[] getNodeValues() {
		//create array to store values
		float[] returnArray = new float[nodeCount];
		//iterate through return array, fill with relevant value
		for(int i = 0; i < nodeCount; ++i) {
			returnArray[i] = getNodeValue(i);
		}
		return returnArray;
	}
	

	public void setNodeValues(float[] values) {
		//loop through each node, ensuring node value is out of bounds of neither values nor nodes
		for(int i = 0; i < values.length && i < nodeCount; ++i)
			setNodeValue(i, values[i]);//set each node value
		if(values.length < nodeCount) {
			System.err.println("ERR 201: Input values do not fill nodes of primary layer.");}
		if(values.length > nodeCount) {
			System.err.println("ERR 200: Input values are greater than the number of nodes in the primary layer.");}
	}
	
	
	
	
	
	//Node Error
	
	public void setNodeError(float[] error) {
		for(int i = 0; i < nodes.length && i < error.length; ++i) {
			setNodeError(i, error[i]);
		}
	}
	
	public void setNodeError(int node, float error) {
		nodes[node].setActivationCost(error);
	}
	
	public float[] getNodeError() {
		float[] error = new float[nodes.length];
		for(int i = 0; i < error.length; ++i) {
			error[i] = nodes[i].getActivationCost();
		}
		return error;
	}
	
	
	
	
	
	
	//Node Activation Cost
	
	public void setNodeActivationCost(int node, float activationcost) {
		nodes[node].setActivationCost(activationcost);
	}
	
	public float getNodeActivationCost(int node) {
		return nodes[node].getActivationCost();
	}
	
	
	public void randomize_nodes(float weightmin, float weightmax, float biasmin, float biasmax, int seed) {
		//iterate through each node in this layer
		for(int i = 0; i < nodeCount; ++i) {
			nodes[i].setBias(RandomGenerator.getFloat()*(biasmax-biasmin) + biasmin);
			//loop through weights 
			for(int j = 0; j < nodes[i].getWeightCount(); ++j) {
				//assign weight random value
				nodes[i].setWeight(j, RandomGenerator.getFloat()*(weightmax-weightmin) + weightmin);
			}
		}
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	/*
	 * Debug console output method
	 */
	
	public void Debug_Print(int node) {
		System.out.print(" N(" + nodes[node].getValue() + ") ");
		for(int i = 0; i < nodes[node].getWeightCount(); ++i) {
			System.out.print( "W"+i+"("+nodes[node].getWeight(i) +") ");
		}
		System.out.print(" B("+nodes[node].getBias()+") ");
	}
	
}
