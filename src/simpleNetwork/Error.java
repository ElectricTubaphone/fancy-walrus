package simpleNetwork;

public final class Error {
	
	
	
	public static float totalOutputError(float[] target, float[] output) {
		float total = 0.f;
		for(int i = 0; i < target.length && i < output.length; ++i) {
			total += .5f*(target[i]-output[i])*(target[i]-output[i]);
		}
		return total;
	}
	
	public static float outputError(float target, float actual) {
		return .5f*(target - actual)*(target-actual);
	}
	
	/*public static dETotal_dOutOutput(float out, float target) {
		return out - target;
	}*/
	
	//dOut/dIn is derivative of activation function
	
	public static float dIn_dWOutput(float outpreviousnode) {
		return outpreviousnode;
	}
}
