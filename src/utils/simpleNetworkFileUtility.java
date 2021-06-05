package utils;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Scanner;

import simpleNetwork.simpleNetwork;

public class simpleNetworkFileUtility {
	 
	
	
	public static void saveFile(simpleNetwork network, String path) {
		IntBuffer header = network.unload_header();
		FloatBuffer body = network.unload();
		
		
		try(BufferedWriter out = new BufferedWriter(new FileWriter(path, false))){
			for(int i = 0; i < header.limit(); ++i) {
				out.write(header.get() + "\n");
			}
			for(int i = 0; i < body.limit(); ++i) {
				out.write(body.get() + "\n");
			}
		} catch (IOException e) {
			System.err.println("IOException error at path \"" + path + "\" ");
		}
	}
	
	public static simpleNetwork loadFile(String path) throws IOException, FileNotFoundException {
		Scanner sc = new Scanner(new FileInputStream(path));
		
		//READ HEADER DATA
		//prepare variables to receive header data
		int layercount = Integer.valueOf(sc.nextLine());
		IntBuffer header = IntBuffer.allocate(2 + layercount);
		header.put(layercount);
		
		//set-up layernodecount to size specified in header
		for(int i = 1; i < layercount + 2; ++i) {
			header.put(Integer.valueOf(sc.nextLine()));
		}
		
		//READ BODY DATA
		//prepare body reading variable
		int floatcount = 0;//record number of floats to load into the FloatBuffer
		for(int i = 0; i < header.get(0)-1; ++i) {
			floatcount += header.get(i + 2) * (header.get(i+3) + 1);//get biases (equal to the number of nodes) plus the number of nodes in the next layer (weight count) for each node in layer
		}
		floatcount += header.get(header.get(0)+1);//get biases for last layer
		FloatBuffer body = FloatBuffer.allocate(floatcount); //TODO: verify allocation size is correct.
		for(int i = 0; i < floatcount; ++i) {
			body.put(Float.valueOf(sc.nextLine()));
		}
		
		header.flip();
		body.flip();
		
		simpleNetwork network = new simpleNetwork(header, body);
		return network;
	}
}
