import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaLibrary;
import com.sun.jna.Native;

import java.io.File;
import java.util.Arrays;
import java.util.regex.Pattern;

public class TestCuda {
    public static void main(String[] args) {
        CudaLibrary LIB = loadLibrary();
        int[] count = new int[1];
        int result = LIB.cudaGetDeviceCount(count);
        System.out.println("++++++++++++++++++++++++++++++++++++++");
        System.out.println("result is: " + result);
        System.out.println("gpu count is: " + count[0]);
        System.out.println("++++++++++++++++++++++++++++++++++++++");
    }

    public static CudaLibrary loadLibrary() {
        try {

            return Native.load("cudart", CudaLibrary.class);
        } catch (UnsatisfiedLinkError e) {
            logger.debug("cudart library not found.");
            logger.trace("", e);
            return null;
        } catch (SecurityException e) {
            logger.warn("Access denied during loading cudart library.");
            logger.trace("", e);
            return null;
        }
    }
}
