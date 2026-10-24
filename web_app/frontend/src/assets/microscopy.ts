export const MICROSCOPY_IMAGES = {
  segmentationColor: "/images/segmentation-color.png",
  comparisonGrey: "/images/comparison-grey.png",
  cellsOutline: "/images/cells-outline.png",
  comparisonGreen: "/images/comparison-green.png",
  fluorescenceRing: "/images/fluorescence-ring.png",
  cellsSemBlue: "/images/cells-sem-blue.png",
  networkClusters: "/images/network-clusters.png",
} as const;

export const SHOWCASE_SLIDES = [
  {
    src: MICROSCOPY_IMAGES.fluorescenceRing,
    label: "Ring nuclei detection",
    sub: "Fluorescence · infected cell ring",
  },
  {
    src: MICROSCOPY_IMAGES.cellsSemBlue,
    label: "SEM morphology scan",
    sub: "High-resolution cell structure",
  },
  {
    src: MICROSCOPY_IMAGES.networkClusters,
    label: "Cell network graph",
    sub: "Spatial relationship mapping",
  },
  {
    src: MICROSCOPY_IMAGES.comparisonGreen,
    label: "Fluorescence segmentation",
    sub: "Original → AI mask",
  },
  {
    src: MICROSCOPY_IMAGES.segmentationColor,
    label: "Multi-class detection",
    sub: "Infected cell mapping",
  },
  {
    src: MICROSCOPY_IMAGES.cellsOutline,
    label: "Phase-contrast analysis",
    sub: "Contour extraction",
  },
  {
    src: MICROSCOPY_IMAGES.comparisonGrey,
    label: "Instance segmentation",
    sub: "Per-cell isolation",
  },
] as const;
