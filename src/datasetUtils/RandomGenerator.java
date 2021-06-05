package datasetUtils;

import java.util.Date;
import java.util.Random;

public class RandomGenerator {
	public static int seed;
	public static Random random = new Random();
	public static int counter = 0;
	
	public RandomGenerator(int seed) {
		this.seed = seed;
		random.setSeed(RandomGenerator.seed * (new Date()).getTime());
	}
	
	public static void SeedGenerator(int seed) {
		RandomGenerator.seed = seed;
		random.setSeed(RandomGenerator.seed * (new Date()).getTime());
	}
	
	public static float getFloat() {
		float out = random.nextFloat();
		++counter;
		if(counter > 10) {
			random.setSeed(random.nextInt());
		}
		return out;
	}
	
	public static float getInt() {
		int out = random.nextInt();
		++counter;
		if(counter > 10) {
			random.setSeed(random.nextInt());
		}
		return out;
	}
	
}
