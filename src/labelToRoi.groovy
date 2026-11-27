import ij.IJ
import ij.ImagePlus
import ij.Prefs
import ij.gui.PolygonRoi
import ij.gui.Roi
import ij.gui.ShapeRoi
import ij.measure.ResultsTable
import ij.plugin.ChannelSplitter
import ij.plugin.ZProjector
import ij.plugin.frame.RoiManager
import loci.plugins.BF
import loci.plugins.in.ImporterOptions
import org.apache.commons.compress.utils.FileNameUtils


// INPUT UI
//
#@File(label = "Input File Directory", style = "directory") inputFilesDir
#@File(label = "Output directory", style = "directory") outputDir
//def inputFilesDir = new File("C:/Users/acayuela/Desktop/users/buyuma/image")
//def outputDir = new File("C:/Users/acayuela/Desktop/users/buyuma/images")
// IDE
//
//
//def headless = true;
//new ij.ImageJ().setVisible(true);
//def inputFilesDir = new File("/mnt/imgserver/CONFOCAL/IA/Projects/2025/2025_1_9_pgonzalez/images/DAB++")
//def outputDir = new File("/mnt/imgserver/CONFOCAL/IA/Projects/2025/2025_1_9_pgonzalez/output/rois/DAB++")
IJ.log("-Parameters selected: ")
IJ.log("    -inputFileDir: " + inputFilesDir)
IJ.log("    -outputDir: " + outputDir)
IJ.log("                                                           ");
/** Get files (images) from input directory */
def listOfFiles = inputFilesDir.listFiles();
def rm = RoiManager.getInstance()
if(rm != null){
    rm.close()
    rm = new RoiManager()
}else{
    rm = new RoiManager()
}
rm.setVisible(false)
for (def i = 0; i < listOfFiles.length; i++) {

//    if (!listOfFiles[i].getName().contains("DS")) {
//        IJ.log("Analyzing file: " + listOfFiles[i].getName());
    /** Define output directory per file */

    def options = new ImporterOptions();
    options.setId(inputFilesDir.getAbsolutePath() + File.separator + listOfFiles[i].getName());
    options.setSplitChannels(false);
    options.setSplitTimepoints(false);
    options.setSplitFocalPlanes(false);
    options.setAutoscale(true);
    options.setStackFormat(ImporterOptions.VIEW_HYPERSTACK);
    options.setStackOrder(ImporterOptions.ORDER_XYCZT);
    options.setColorMode(ImporterOptions.COLOR_MODE_COMPOSITE);
    options.setCrop(false);
    options.setOpenAllSeries(true);
    def imps = BF.openImagePlus(options);
//
//

    def imp = imps[0]
    imp = new ImagePlus("",imp.getStack().getProcessor(2))
    //IJ.saveAs(imp,"Tiff", outputDir.getAbsolutePath()+File.separator+listOfFiles[i].getName())
    imp.show()
    IJ.setAutoThreshold(imp, "Default dark");
    Prefs.blackBackground = true;
    IJ.run(imp, "Convert to Mask", "");
    IJ.run(imp, "Create Selection", "");
    def roi = imp.getRoi()
    rm.addRoi(roi)
    rm.select(0)
    rm.runCommand("Save",outputDir.getAbsolutePath() + File.separator +listOfFiles[i].getName().replaceAll(".tif",".roi"))
    rm.reset()
    imp.hide()


    //}
}