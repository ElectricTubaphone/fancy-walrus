package simpleNetwork;

class Node {
	private float[] weight;
	private float weightCount;
	private float bias;
	private float biasChange;
	private float value;
	private float input;
	private float[] weightChange;
	private float error;
	private float activationCost;
	
	public Node(int nextLayerSize) {
		weight = new float[nextLayerSize];
		weightCount = nextLayerSize;
		weightChange = new float[nextLayerSize];
	}
	
	public void setActivationCost(float activationcost) {
		activationCost = activationcost;
	}
	
	public float getActivationCost() {
		return activationCost;
	}
	
	public float getError() {
		return error;
	}
	
	public void setError(float error) {
		this.error = error;
	}
	
	public void setBiasChange(float error) {
		biasChange = error;
	}
	
	
	public void applyWeightChange(float learningrate) {
		for(int i = 0; i < weightCount; ++i)
			if(getWeightChange(i) != 0 && getWeightChange(i) != Float.NaN) {weight[i] -= learningrate * getWeightChange(i);}
	}
	
	public void applyBiasChange(float learningrate) {
		bias -= learningrate * getBiasChange();
	}
	
	
	public void setWeightChange(int weight, float error) {
		this.weightChange[weight] = error;
	}
	
	
	public float getBiasChange() {
		return biasChange;
	}
	
	public float getWeightChange(int weight) {
		return weightChange[weight];
	}
	
	public void setInput(float input) {
		this.input = input;
	}
	
	public void setValue(float val) {
		value = val;
	}
	
	public void setWeight(int index, float weight) {
		this.weight[index] = weight;
	}
	
	public void setBias(float bias) {
		this.bias = bias;
	}
	
	public float getInput() { 
		return input;
	}
	
	public float getValue() {
		return value;
	}
	
	public float getBias() {
		return bias;
	}
	
	public float getWeight(int nextnode) {
		return weight[nextnode];
	}
	
	public float getWeightCount() {
		return weightCount;
	}
}
