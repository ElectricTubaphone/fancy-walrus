package simpleNetwork;

public final class Activation {
	
	public enum ACTIVATION {
		SIGMOID, TANH, RELU, LEAKY_RELU
	};
	
	
	/*
	 * Sigmoid
	 */
	
	public static float sigmoid(float input) {
		return 1.f/(1.f+(float)Math.exp(-input));
	}
	
	public static float sigmoid_derivative(float input) {
		return sigmoid(input) * ( 1.f - sigmoid(input));
	}
	
	
	
	
	
	
	
	
	/*
	 * Hyperbolic Tangent
	 */
	
	public static float tanh(float input) {
		return ((float) (Math.exp(input) - Math.exp(-input))/((float)(Math.exp(input) + Math.exp(-input))));
	}
	
	public static float tanh_derivative(float input) {
		return 1.f-(float)Math.pow(Math.tanh(input), 2);
	}
	
	
	
	
	
	
	
	
	
	/*
	 * Rectified Linear Activation Units (ReLU)
	 */
	
	public static float ReLU(float input) {
		return max(0, input);
	}
	
	public static float ReLU_derivative(float input) {
		return input < 0 ? 0 : 1;
	}
	
	
	
	
	
	
	
	/*
	 * Leaky ReLU
	 */
	
	public static float leaky_ReLU(float input) {
		return max(0.01f * input, input);
	}
	
	public static float leaky_ReLU_derivative(float input) {
		return input < 0 ? .01f : 1.f;
	}
	
	
	
	
	
	
	
	
	//Additional utility method for ReLU
	
	private static float max(float a, float b) {
		return a > b ? a : b;
	}
}
