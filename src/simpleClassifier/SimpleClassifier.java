package simpleClassifier;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import simpleNetwork.Activation;
import simpleNetwork.Activation.ACTIVATION;
import utils.simpleNetworkFileUtility;
import simpleNetwork.simpleNetwork;

public class SimpleClassifier {
	private simpleNetwork network;
	String[] terms;
	
	public SimpleClassifier(simpleNetwork network) {
		this.network = network;
	}
	
	public SimpleClassifier(IntBuffer header, FloatBuffer body, String[] term) {
		network = new simpleNetwork(header, body);
		terms = term;
	}
	
	public SimpleClassifier(IntBuffer header, FloatBuffer body) {
		network = new simpleNetwork(header, body);
	}
	
	public SimpleClassifier(int[] layernodecount, Activation.ACTIVATION activation, String[] definitions) {
		network = new simpleNetwork(layernodecount, activation);
		terms = new String[layernodecount[layernodecount.length-1]];//set number of classification terms equal to number of classifier outputs
		defineTerms(definitions);
	}
	
	public SimpleClassifier(int[] layernodecount, ACTIVATION activation, float wmin, float wmax, float bmin, float bmax, float learningRate) {
		network = new simpleNetwork(layernodecount, activation, wmin, wmax, bmin, bmax, learningRate);
	}
	
	public SimpleClassifier(String path, float learningRate) {
		try {
			network = simpleNetworkFileUtility.loadFile(path);
		} catch (FileNotFoundException e){
			System.err.println("Error 300: FileNotFoundException: Could not load network from path \'" + path + "\'.");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.err.println("Error 301: IOException: Could not load network from path \'" + path + "\'.");
		}
	}
	
	public simpleNetwork getNetwork() {
		return network;
	}
	
	public void saveNetwork() {
		simpleNetworkFileUtility.saveFile(network, "C:\\Users\\User\\Downloads\\Java\\simpleNN\\network.cfg");
	}
	
	public IntBuffer unloadHeader() {
		return network.unload_header();
	}
	
	public FloatBuffer unloadBody() {
		return network.unload();
	}
	
	public String[] unloadTerms() {
		return terms;
		
	}
	
	public void defineTerm(int value, String term) {
		if(value < terms.length) {
			terms[value] = term;
		}
	}
	
	public void defineTerms(String[] term) {
		for(int i = 0; i < terms.length && i < term.length; ++i) {
			terms[i] = term[i];
		}
	}
	
	private int getTermsIndex(String term) {
		int i = 0;
		while(i < terms.length && terms[i] != term) {++i;}
		if(i < terms.length ) {
			return i;
		}
		else
		{
			return -1;
		}
	}
	
	public int getClassificationCount() {
		return network.getOutputValues().length;
	}
	
	public float[] getOutputValues() {
		return network.getOutputValues();
	}
	
	public void setLearningRate(float rate) {
		network.setLearningRate(rate);
	}
	
	public int propagate(float[] input) {
		network.setInputValues(input);
		network.propagate();
		float[] output = network.getOutputValues();
		int i = 0;
		float height = output[0];
		for(int j = 0; j < output.length; ++j) {
			if(output[j] > height) {
				i = j; height = output[j];
			}
		}
		return i;
	}
	
	public String propagate_str(float[] input) {
		return terms[propagate(input)];
	}
	
	//run each provided piece of data into network.training in order
		public void train(float[][] trainingset, String[] classifications) {
			//get a blank output float array, to later mark with 1.0f at classified values
			float[] zeroed_output = new float[network.getOutputValues().length];
			for(int i = 0 ; i < network.getOutputValues().length; ++i) {
				zeroed_output[i] = 0.f;//zero each value of array
			}
			
			//create target array to use to compare classifier output
			float[] output_target = new float[network.getOutputValues().length];
			int index;
			for(int i = 0; i < trainingset.length && i < classifications.length; ++i) {
				//get location value of node corresponding to the string representing the apposite classification
				index = getTermsIndex(classifications[i]);
				output_target = zeroed_output;
				//if no value is classified for(i.e. terms contains no such string), then have all values be 0. Else, set output_target[index] to 1
				if(index >= 0 && index < output_target.length)
				{
					//indicate the desired classification
					output_target[index] = 1.f;
				}
				//now propagate and backpropagate based on input, modify weights and biases
				network.train(trainingset[i], output_target);
			}
		}
	
	//run each provided piece of data into network.training in order
	public void train(float[][] trainingset, int[] classifications) {
		//get a blank output float array, to later mark with 1.0f at classified values
		float[] zeroed_output = new float[network.getOutputValues().length];
		for(int i = 0 ; i < network.getOutputValues().length; ++i) {
			zeroed_output[i] = 0.f;//zero each value of array
		}
		
		//create target array to use to compare classifier output
		float[] output_target = new float[network.getOutputValues().length];
		for(int i = 0; i < trainingset.length && i < classifications.length; ++i) {
			output_target = zeroed_output;
			//indicate the desired classification
			output_target[classifications[i]] = 1.f;
			//now propagate and backpropagate based on input, modify weights and biases
			network.train(trainingset[i], output_target);
		}
	}
	
	
	public void train(float[] trainingdata, int classification) {
		//get a blank output float array, to later mark with 1.0f at classified values
		float[] zeroed_output = new float[network.getOutputValues().length];
		for(int i = 0 ; i < network.getOutputValues().length; ++i) {
			zeroed_output[i] = 0.f;//zero each value of array
		}
		
		//create target array to use to compare classifier output
		float[] output_target = new float[network.getOutputValues().length];
		
		output_target = zeroed_output;
		//indicate the desired classification
		output_target[classification] = 1.f;
		//now propagate and backpropagate based on input, modify weights and biases
		network.train(trainingdata, output_target);
	}
}
