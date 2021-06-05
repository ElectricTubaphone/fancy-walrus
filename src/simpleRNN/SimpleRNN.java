package simpleRNN;

import simpleNetwork.Activation;
import simpleNetwork.simpleNetwork;

public class SimpleRNN {
	private simpleNetwork network;
	float[] outputCache;
	
	public SimpleRNN(int[] layers, Activation.ACTIVATION activationFunction, float wmin, float wmax, float bmin, float bmax, float LR) {
		//for the specification, layers[0] indicates the number of additional layers past the first recurrent entry (equal to the number of output nodes)
		network = new simpleNetwork(layers, activationFunction, wmin, wmax, bmin, bmax, LR);
		outputCache = new float[network.getOutputValues().length];
	}
	
	public void setOutputCache(float[] cache) {
		for(int i = 0; i < outputCache.length && i < cache.length; ++i) {
			outputCache[i] = cache[i];
		}
	}
	
	public float[] getOutputCache() {
		return outputCache;
	}
	
	public float[] propagate(float[] recurrentEntry, float[] generalInput)
	{
		float[] input = concatenate(recurrentEntry, generalInput);
		network.propagate(input);
		outputCache = network.getOutputValues();
		return network.getOutputValues();
	}
	
	public float[] propagate(float[] generalInput) {
		float[] input = concatenate(outputCache, generalInput);
		network.propagate(input);
		outputCache = network.getOutputValues();
		return network.getOutputValues();
	}
	
	public void trainLoop(float[] input, float[] target) {
		network.train(concatenate(outputCache, input), target);
	}
	
	public void train(float[] cache, float[] input, float[] target) {
		network.train(concatenate(cache, input), target);
	}
	
	public void trainIterate(float[] cache, float[][] input, float[][] target) {
		setOutputCache(cache);
		for(int i = 1; i < input.length && i < target.length; ++i) 
		{
			network.train(concatenate(outputCache, input[i]), target[i]);
		}
	}
	
	
	public void backpropagate(float target[]) {
		network.backpropagateTarget(target);
	}
	
	public void backpropagate(float target[], float learningRate) {
		network.backpropagateTarget(target, learningRate);
	}

	public void backpropagateLoop() {
		network.backpropagateLoop();
	}
	
	public float[] concatenate(float[] a, float[] b) {
		float[] c = new float[a.length + b.length];
		for(int i = 0; i < a.length; ++i) {
			c[i] = a[i];
		}
		for(int i = a.length; i < a.length + b.length; ++i) {
			c[i] = b[i - a.length];
		}
		return c;
	}
}
