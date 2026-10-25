## rumz 8/25

import matplotlib.pyplot as plt

def Display(img, wvls, step="", outputPath="", save=False, note=""):
    if note != "":
        readme_path = outputPath / "README.md"
        with readme_path.open("a") as f:
                f.write(f"\n######  DEBUG STEP {step}: {note}\n")
    for band in range(img.shape[2]):
        plt.figure(figsize=(10, 15))
        plt.title(f"Wavelength: {wvls[band]}nm")
        plt.imshow(img[:,:,band], cmap='gray')
        if save == True:
            filename = f"Debug{step}_Band{band}.png"
            readme_path = outputPath / "README.md"
            with readme_path.open("a") as f:
                f.write(f"\n###  Output File: {filename}\n- Debug Step:{step}\n")
            plt.savefig(outputPath / filename)
        plt.show()
        