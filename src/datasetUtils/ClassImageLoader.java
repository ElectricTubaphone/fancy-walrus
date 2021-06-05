package datasetUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;



public class ClassImageLoader {
	
	List<String> container_pathlist;
	
	
	public ClassImageLoader(String[] paths) {
		container_pathlist = new ArrayList<String>();
		for(String path:paths) {
			container_pathlist.add(path);
		}
	}
	
	public ClassImageLoader() {
		container_pathlist = new ArrayList<String>();
	}
	
	public void addPath(String path) {
		container_pathlist.add(path);
	}
	
	public void removePath(int pathID) {
		container_pathlist.remove(pathID);
	}
	
	public String[] getPaths() {
		return container_pathlist.toArray(new String[container_pathlist.size()]);
	}
	
	public File getRandomFile(int containerID) {
		File dir = new File(container_pathlist.get(containerID));
		File[] candidates = dir.listFiles();
		File file = candidates[(new Random()).nextInt(candidates.length)];
		return file;
	}
	
	public BufferedImage getRandomBufferedImage(int containerID) {
		File file = getRandomFile(containerID);
		BufferedImage img;
		try {
			img = ImageIO.read(file);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.err.println("ImageIO.read(file) IOException");
			img = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
		}
		//may return a blank image with unintended value
		return img;
	}
	
	public float[] getRandomImageSequence(int containerID) {
		BufferedImage img = getRandomBufferedImage(containerID);
		float[] data = new float[img.getWidth() * img.getHeight() * 3];
		Color colorGetter;
		for(int i = 0; i < img.getHeight(); ++i) {
			for(int j = 0; j < img.getWidth() * 3; j += 3) {
				colorGetter = new Color(img.getRGB(j/3, i));
				data[j + i*img.getWidth()*3] = ((float)colorGetter.getRed())/255.f;
				data[j + 1 + i*img.getWidth()*3] = ((float)colorGetter.getGreen())/255.f;
				data[j + 2 + i*img.getWidth()*3] = ((float)colorGetter.getBlue())/255.f;
			}
		}
		return data;
	}
	
}
