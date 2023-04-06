import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class KMeans {
	public static void main(String[] args) {
		if (args.length < 3) {
			System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
			return;
		}

		try {
			BufferedImage originalImage = ImageIO.read(new File(args[0]));
			int k = Integer.parseInt(args[1]);
			BufferedImage kmeansJpg = kmeans_helper(originalImage, k);
			ImageIO.write(kmeansJpg, "jpg", new File(args[2]));
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}
	}

	private static BufferedImage kmeans_helper(BufferedImage originalImage, int k) {
		int w = originalImage.getWidth();
		int h = originalImage.getHeight();
		BufferedImage kmeansImage = new BufferedImage(w, h, originalImage.getType());
		Graphics2D g = kmeansImage.createGraphics();
		g.drawImage(originalImage, 0, 0, w, h, null);
		// Read rgb values from the image
		int[] rgb = new int[w * h];
		int count = 0;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				rgb[count++] = kmeansImage.getRGB(i, j);
			}
		}
		// Call kmeans algorithm: update the rgb values
		kmeans(rgb, k);

		// Write the new rgb values to the image
		count = 0;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				kmeansImage.setRGB(i, j, rgb[count++]);
			}
		}
		return kmeansImage;
	}

	/**
	 * Update the array rgb by assigning each entry in the rgb array to its cluster
	 * center
	 */
	private static void kmeans(int[] rgb, int k) {
		// maps each pixel to its designted cluster
		final int rgbCluster[] = new int[rgb.length];
		// store the pixel value of the cluster centers
		final int clusterCenters[] = new int[k];

		// randomly initialize the clusters
		for (int i = 0; i < k; i++) {
			clusterCenters[i] = rgb[(int) (Math.random() * rgb.length)];
		}

		// run the centroid update step until a negligable change
		int centroid_delta = Integer.MAX_VALUE;
		while (centroid_delta > 0) {
			// place each rgb value into its closest cluster
			for (int i = 0; i < rgb.length; i++) {
				rgbCluster[i] = computeCluster(clusterCenters, rgb[i]);
			}

			// compute the new cluster centroid rgb values
			final int sum[] = new int[k];
			final int count[] = new int[k];
			for (int i = 0; i < rgb.length; i++) {
				final int clusterIndex = rgbCluster[i];
				sum[clusterIndex] += rgb[i];
				count[clusterIndex]++;
			}
			final int tempClusterCenters[] = new int[k];
			for (int i = 0; i < k; k++) {
				// check for divide by 0 cases
				tempClusterCenters[i] = sum[i] / count[i];
			}

			// compute the change between the old and new cluster center values
			centroid_delta = 0;
			for (int i = 0; i < k; k++) {
				centroid_delta += Math.abs(tempClusterCenters[i] - clusterCenters[i]);
			}

			// update with the new cluster values
			for (int i = 0; i < k; k++) {
				clusterCenters[i] = tempClusterCenters[i];
			}
		}

		// update each index of the rgb array with the centroid
		// value of its corresponding cluster
		for (int i = 0; i < rgb.length; i++) {
			rgb[i] = clusterCenters[rgbCluster[i]];
		}
	}

	/**
	 * Compute the Cluster which a pixel belongs to based on squared RGB distance
	 * 
	 * @param centers list of rgb values for the centroids
	 * @param pixel   the rgb value to place in a cluster
	 * @return the index of the cluster that the pixel belongs to
	 */
	private static int computeCluster(int[] centers, int pixel) {
		int minDistance = Integer.MAX_VALUE;
		int minDistanceIndex = Integer.MAX_VALUE;
		for (int i = 0; i < centers.length; i++) {
			final int distance = computeRGBDistance(pixel, centers[i]);
			if (distance < minDistance) {
				minDistance = distance;
				minDistanceIndex = i;
			}
		}
		return minDistanceIndex;
	}

	/**
	 * Compute the difference between two rgb values using the squared
	 * difference between each respective 8-bit pixel value
	 * 
	 * @return an int representing squared distance between two pixel values
	 */
	private static int computeRGBDistance(final int rgb1, final int rgb2) {
		// use & 0xff to get the least-significant 8 bits from an integer
		// and shift the >> shift operator to get to the desired rgb offset
		final int dr = ((rgb1 >> 16) & 0xff) - ((rgb2 >> 16) & 0xff);
		final int dg = ((rgb1 >> 8) & 0xff) - ((rgb2 >> 8) & 0xff);
		final int db = (rgb1 & 0xff) - (rgb2 & 0xff);
		return dr * dr + dg * dg + db * db;
	}
}
