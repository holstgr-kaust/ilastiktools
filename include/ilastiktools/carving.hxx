#ifndef VIGRA_ILASTIKTOOLS_CARVING_HXX
#define VIGRA_ILASTIKTOOLS_CARVING_HXX

#include <vigra/adjacency_list_graph.hxx>
#include <vigra/timing.hxx>
#include <vigra/multi_gridgraph.hxx>
#include <vigra/timing.hxx>
#include <vigra/graph_algorithms.hxx>

#include <assert.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif 


namespace
{
/**
 * @brief Validate region of interest for block shape
 *
 * Asserts that roiEnd is within given shape.
 * For DEBUG builds only.
 *
 * @param blockShape
 * @param roiEnd
 */
template<unsigned int DIM>
void validateRegionShape(
          const vigra::TinyVector<vigra::MultiArrayIndex, DIM>& blockShape
        , const vigra::TinyVector<vigra::MultiArrayIndex, DIM>& roiEnd)
{
    for (unsigned int i = 0; i<DIM; ++i)
    {
        assert( roiEnd[i] <= blockShape[i] && "Expected roi inside array" );
    }
}

/**
 * @brief Validate block shapes are equal
 *
 * For DEBUG builds only.
 *
 * @param shape1
 * @param shape2
 */
template<unsigned int DIM>
void validateEqualShapes(
          const vigra::TinyVector<vigra::MultiArrayIndex, DIM>& shape1
        , const vigra::TinyVector<vigra::MultiArrayIndex, DIM>& shape2)
{
    for (unsigned int i = 0; i<DIM; ++i)
    {
        assert( shape1[i] == shape2[i] && "Expected matching shapes" );
    }
}
}

namespace vigra
{
    template<unsigned int DIM, class LABELS>
    class GridRag : public AdjacencyListGraph
    {
        #ifdef WITH_OPENMP
        // NOTE: OMP_EDGE_LOCKS must equal (2^x)-1; i.e., it must be a mask and a (count - 1).
        // TODO: NOTE: Encountered issue where all threads were suspended with large OMP_EDGE_LOCKS
        //             Suspect thread contention between OMP in ilastiktools, and threads in lazyflow;
        //             It works with a smaller lock pool.
        static const std::size_t OMP_EDGE_LOCKS = 0x7F; //2^7-1 //0xFFF; // 2^12-1
        #endif
    public:
        typedef GridGraph<DIM, boost_graph::undirected_tag>  GridGraphType;
        typedef LABELS LabelType;
        typedef TinyVector<MultiArrayIndex, DIM>  ShapeN;
        typedef TinyVector<MultiArrayIndex,   1>  Shape1;

        GridRag() : AdjacencyListGraph()
        {
            #ifdef WITH_OPENMP
            edgeLocks_ = new omp_lock_t[OMP_EDGE_LOCKS + 1];
            #pragma omp parallel for
            for(size_t i=0; i<OMP_EDGE_LOCKS;++i)
            {
              omp_init_lock(&(edgeLocks_[i]));
            }
            #endif
        }

        ~GridRag()
        {
            #ifdef WITH_OPENMP
            #pragma omp parallel for
            for(size_t i=0; i<OMP_EDGE_LOCKS;++i)
            {
                omp_destroy_lock(&(edgeLocks_[i]));
            }
            delete[] edgeLocks_;
            #endif
        }

        /**
         * @brief Find edge id between given labels in graph.
         * @param lu
         * @param lv
         */
        inline index_type findEdgeFromIds(const LabelType lu, const LabelType lv)
        {
            const Edge e  = findEdge(nodeFromId(lu), nodeFromId(lv));
            return id(e);
        }

        void assignLabels(
                  const MultiArrayView<DIM, LABELS>& labels
                , const ShapeN roiEnd)
        {
#ifndef NDEBUG
            validateRegionShape<DIM>(labels.shape(), roiEnd);
#endif

            LABELS minLabel, maxLabel;
            labels.minmax(&minLabel, &maxLabel);

            growNodeRange(maxLabel);

            addEdges(labels, roiEnd);
        }

        void assignLabelsFromSerialization(
            const MultiArrayView<1, LABELS>& serialization )
        {
            deserialize(serialization.begin(), serialization.end());
        }


        /**
         * @brief Accumulate edge features
         *
         * Edge feature accumulation is a stage that creates the edge-wise
         * feature weights from input labels and feature arrays.  It is designed
         * to support piece-wise accumulation provided the following rules
         * are satisfied:
         *
         * labels and featuresIn (henceforth arrays) are the same dimensions,
         * The arrays represent the same position in the whole,
         * A block is the space from the start of the array to its roiEnd,
         * Non-boundary blocks must include a halo of size 1 on their end regions,
         *  (e.g., roiEnd[i] < array.shape()[i] for non-edge pieces).
         * accumulate is called for all blocks such that each point in the
         * original space is represented inside an array block once, and only once.
         *
         * @param labels Array of per-point label ids.
         * @param featuresIn Array of per-point feature values.
         * @param roiEnd Specifies the block boundary on the upper side;
         *          equals array size for blocks on the boundaries of the
         *          original space, is smaller than the array size otherwise.
         * @param featuresOut A per-edge list of total accumulated weights
         * @param featureCountsOut A per-edge list of accumulation counts
         *          (used to calculate weight averages).
         */
        template<class WEIGHTS_IN, class WEIGHTS_OUT>
        void accumulateEdgeFeatures(
              const MultiArrayView<DIM, LABELS>& labels
            , const MultiArrayView<DIM, WEIGHTS_IN>& featuresIn
            , const ShapeN roiEnd
            , MultiArrayView<1, WEIGHTS_OUT >& featuresOut
            , MultiArrayView<1, UInt32>& featureCountsOut)
        {
#ifndef NDEBUG
            validateRegionShape<DIM>(labels.shape(), roiEnd);
            validateRegionShape<DIM>(featuresIn.shape(), roiEnd);
            assert(featuresOut.size() == edgeNum());
            assert(featureCountsOut.size() == edgeNum());
#endif

            calcFeatures( labels, featuresIn, roiEnd
                        , featuresOut, featureCountsOut );
        }

    private:

        /**
         * @brief Add edges in labels array to graph
         *
         * @param labels A contiguous block of labels, possibly with halo at end.
         * @param roiEnd Specifies end of labels region to insert, in labels coords.
         */
        template<unsigned int NDIM>
        void addEdges( const MultiArrayView<NDIM, LABELS>& labels
                     , const ShapeN& roiEnd)
        {
            throw std::runtime_error("Currently only 2D and 3D is supported");
        }

        // addEdges<2>
        void addEdges( const MultiArrayView<2, LABELS>& labels
                     , const ShapeN& roiEnd)
        {
            const ShapeN shape = labels.shape();
            for(MultiArrayIndex y=0; y<roiEnd[1]; ++y)
            for(MultiArrayIndex x=0; x<roiEnd[0]; ++x)
            {
                const LabelType l  = labels(x, y);
                if(x+1 < shape[0])
                    maybeAddEdge(l, labels(x+1, y));
                if(y+1 < shape[1])
                    maybeAddEdge(l, labels(x, y+1));
            }
        }

        // addEdges<3>
        void addEdges( const MultiArrayView<3, LABELS>& labels
                     , const ShapeN& roiEnd)
        {
            const ShapeN shape = labels.shape();
            for(MultiArrayIndex z=0; z<roiEnd[2]; ++z)
            for(MultiArrayIndex y=0; y<roiEnd[1]; ++y)
            for(MultiArrayIndex x=0; x<roiEnd[0]; ++x)
            {
                const LabelType l  = labels(x, y, z);
                if(x+1 < shape[0])
                    maybeAddEdge(l, labels(x+1, y, z));
                if(y+1 < shape[1])
                    maybeAddEdge(l, labels(x, y+1, z));
                if(z+1 < shape[2])
                    maybeAddEdge(l, labels(x, y, z+1));
            }
        }

        /**
         * @brief Calculate edge-wise features from labels and input features.
         *
         * @param labels A contiguous block of labels, possibly with halo at end.
         * @param featuresIn A contiguous block of features covering same range as labels.
         * @param roiEnd Specifies end of labels / featuresIn region to insert,
         *          in labels / featuresIn coords.
         * @param featuresOut An array of edge-wise sums of features.
         * @param featureCountsOut An array of edge-wise feature counts.
         */
        template<unsigned int NDIM, class WEIGHTS_IN, class WEIGHTS_OUT>
        void calcFeatures(
              const MultiArrayView<NDIM, LABELS>& labels
            , const MultiArrayView<NDIM, WEIGHTS_IN>& featuresIn
            , const ShapeN roiEnd
            , MultiArrayView<1, WEIGHTS_OUT >& featuresOut
            , MultiArrayView<1, UInt32>& featureCountsOut)
        {
            throw std::runtime_error("Currently only 2D and 3D is supported");
        }

        // calcFeatures<2>
        template<class WEIGHTS_IN, class WEIGHTS_OUT>
        void calcFeatures(
              const MultiArrayView<2, LABELS>& labels
            , const MultiArrayView<2, WEIGHTS_IN>& featuresIn
            , const ShapeN roiEnd
            , MultiArrayView<1, WEIGHTS_OUT >& featuresOut
            , MultiArrayView<1, UInt32>& featureCountsOut)
        {
            const ShapeN shape = labels.shape();

            //do the accumulation
            for(MultiArrayIndex y=0; y<roiEnd[1]; ++y)
            for(MultiArrayIndex x=0; x<roiEnd[0]; ++x)
            {
                const LabelType lu  = labels(x, y);

                if(x+1 < shape[0])
                {
                    const LabelType lv = labels(x+1, y);
                    if(lu!=lv)
                    {
                        const index_type eid = findEdgeFromIds(lu, lv);
                        featureCountsOut[eid] += 2;
                        featuresOut[eid] += static_cast<WEIGHTS_OUT>(featuresIn(x,y))
                                          + static_cast<WEIGHTS_OUT>(featuresIn(x+1,y));
                    }
                }

                if(y+1 < shape[1])
                {
                    const LabelType lv = labels(x, y+1);
                    if(lu!=lv)
                    {
                        const index_type eid = findEdgeFromIds(lu, lv);
                        featureCountsOut[eid] += 2;
                        featuresOut[eid] += static_cast<WEIGHTS_OUT>(featuresIn(x,y))
                                          + static_cast<WEIGHTS_OUT>(featuresIn(x,y+1));
                    }
                }
            }
        }

        // calcFeatures<3>
        template<class WEIGHTS_IN, class WEIGHTS_OUT>
        void calcFeatures(
              const MultiArrayView<3, LABELS>& labels
            , const MultiArrayView<3, WEIGHTS_IN>& featuresIn
            , const ShapeN roiEnd
            , MultiArrayView<1, WEIGHTS_OUT >& featuresOut
            , MultiArrayView<1, UInt32>& featureCountsOut)
        {
            const ShapeN shape = labels.shape();

            //do the accumulation
            #ifdef WITH_OPENMP
            #pragma omp parallel for
            #endif
            for(MultiArrayIndex z=0; z<roiEnd[2]; ++z)
            for(MultiArrayIndex y=0; y<roiEnd[1]; ++y)
            for(MultiArrayIndex x=0; x<roiEnd[0]; ++x)
            {
                const LabelType lu  = labels(x, y, z);

                if(x+1 < shape[0])
                {
                    const LabelType lv = labels(x+1, y, z);
                    if(lu!=lv)
                    {
                        const index_type eid = findEdgeFromIds(lu, lv);
                        #ifdef WITH_OPENMP
                        omp_set_lock(&(edgeLocks_[eid & OMP_EDGE_LOCKS]));
                        #endif
                        featureCountsOut[eid] += 2;
                        featuresOut[eid] += static_cast<WEIGHTS_OUT>(featuresIn(x,y,z))
                                          + static_cast<WEIGHTS_OUT>(featuresIn(x+1,y,z));
                        #ifdef WITH_OPENMP
                        omp_unset_lock(&(edgeLocks_[eid & OMP_EDGE_LOCKS]));
                        #endif
                    }
                }

                if(y+1 < shape[1])
                {
                    const LabelType lv = labels(x, y+1, z);
                    if(lu!=lv)
                    {
                        const index_type eid = findEdgeFromIds(lu, lv);
                        #ifdef WITH_OPENMP
                        omp_set_lock(&(edgeLocks_[eid & OMP_EDGE_LOCKS]));
                        #endif
                        featureCountsOut[eid] += 2;
                        featuresOut[eid] += static_cast<WEIGHTS_OUT>(featuresIn(x,y,z))
                                          + static_cast<WEIGHTS_OUT>(featuresIn(x,y+1,z));
                        #ifdef WITH_OPENMP
                        omp_unset_lock(&(edgeLocks_[eid & OMP_EDGE_LOCKS]));
                        #endif
                    }
                }

                if(z+1 < shape[2])
                {
                    const LabelType lv = labels(x, y, z+1);
                    if(lu!=lv)
                    {
                        const index_type eid = findEdgeFromIds(lu, lv);
                        #ifdef WITH_OPENMP
                        omp_set_lock(&(edgeLocks_[eid & OMP_EDGE_LOCKS]));
                        #endif
                        featureCountsOut[eid] += 2;
                        featuresOut[eid] += static_cast<WEIGHTS_OUT>(featuresIn(x,y,z))
                                          + static_cast<WEIGHTS_OUT>(featuresIn(x,y,z+1));
                        #ifdef WITH_OPENMP
                        omp_unset_lock(&(edgeLocks_[eid & OMP_EDGE_LOCKS]));
                        #endif
                    }
                }
            }
        }

        /**
         * @brief add label edges to graph is they are different.
         * @param lu
         * @param lv
         */
        inline void maybeAddEdge(const LabelType lu, const LabelType lv)
        {
            if(lu != lv)
            {
                addEdge( nodeFromId(lu), nodeFromId(lv));
            }
        }

        /**
         * @brief Grow the maximum node range, if necessary.
         * @param maxLabel The current max label id to include.
         */
        void growNodeRange(LABELS maxLabel)
        {
            for (LABELS id = nodeNum(); id <= maxLabel; ++id)
            {
                addNode(id);
            }
        }

    private:
    #ifdef WITH_OPENMP
        omp_lock_t* edgeLocks_;
    #endif
    };

    template<class T>
    struct GridSegmentorEdgeMap
    {
        typedef T Value;
        typedef T& Reference;
        typedef const T& ConstReference;

        GridSegmentorEdgeMap(MultiArrayView<1, T>& values)
        : values_(values)
        {
        }

        template<class K>
        inline Reference operator[](const K key)
        {
            return values_[key.id()];
        }

        template<class K>
        inline ConstReference operator[](const K key)const
        {
            return values_[key.id()];
        }

        MultiArrayView<1, T> values_;
    };

    template<class T>
    struct GridSegmentorNodeMap
    {
        typedef T Value;
        typedef T& Reference;
        typedef const T& ConstReference;

        GridSegmentorNodeMap(MultiArrayView<1, T>& values)
        : values_(values)
        {
        }

        template<class K>
        inline Reference operator[](const K key)
        {
            return values_[key.id()];
        }

        template<class K>
        inline ConstReference operator[](const K key)const
        {
            return values_[key.id()];
        }

        MultiArrayView<1, T>& values_;
    };

    template< unsigned int DIM
            , class LABELS
            , class VALUE_TYPE>
    class GridSegmentor
    {
    public:
        typedef GridRag<DIM, LABELS> Graph;
        typedef TinyVector<MultiArrayIndex, 1>   Shape1;
        typedef TinyVector<MultiArrayIndex, DIM> ShapeN;

        typedef UInt8 SegmentType;

        typedef MultiArrayView<DIM, LABELS> LabelView;
        typedef typename LabelView::const_iterator  LabelViewIter;

        GridSegmentor()
        : graph_()
        , edgeWeights_()
        , edgeCounts_()
        , nodeSeeds_()
        , resultSegmentation_()
        , isFinalized_(false)
        {
        }

        /**
         * @brief Initialize
         *
         * Initialize segmentor.
         */
        void init()
        {
            graph_.clear();

            edgeWeights_.reshape(Shape1(0));
            edgeCounts_.reshape(Shape1(0));
            nodeSeeds_.reshape(Shape1(0));
            resultSegmentation_.reshape(Shape1(0));

            isFinalized_ = false;
        }

        /**
         * @brief Initialize from serialization data
         *
         * Initialize segmentor from previous pre-processed values.
         *
         * @param serialization Label nodes list
         * @param edgeWeights Edge weights list
         * @param nodeSeeds Segmentation seeds
         * @param resultSegmentation Per-node segmentation label
         */
        void initFromSerialization(
              const MultiArrayView< 1, LABELS>& serialization
            , const MultiArrayView< 1, VALUE_TYPE>& edgeWeights
            , const MultiArrayView< 1, SegmentType>& nodeSeeds
            , const MultiArrayView< 1, SegmentType>& resultSegmentation )
        {
            //USETICTOC;

            //TIC;
            // get the RAG
            graph_.assignLabelsFromSerialization(serialization);
            //TOC;

            // Assign weights and seeds and resultSegmentation
            edgeWeights_ = edgeWeights;
            edgeCounts_.reshape(Shape1(0));
            nodeSeeds_ = nodeSeeds;
            resultSegmentation_ = resultSegmentation;

            isFinalized_ = true;
        }

        /**
         * @brief Preprocessing step
         *
         * Preprocessing is a stage that creates the segmentation graph from
         * input labels and weight arrays.  It is designed to support piece-wise
         * pre-preprocessing provided the following rules are satisfied:
         *
         * labels and weightArray (henceforth arrays) are the same dimensions,
         * The arrays represent the same position in the whole,
         * A block is the space from the start of the array to its roiEnd,
         * Non-boundary blocks must include a halo of size 1 on their end regions,
         *  (e.g., roiEnd[i] < array.shape()[i] for non-edge pieces).
         * preprocessing is called for all blocks such that each point in the
         * original space is represented inside an array block once, and only once.
         *
         * @param labels Array of per-point label ids.
         * @param weightArray Array of per-point feature values.
         * @param roiEnd Specifies the block boundary on the upper side;
         *          equals array size for blocks on the boundaries of the
         *          original space, is smaller than the array size otherwise.
         */
        template<class WEIGHTS_IN>
        void preprocessing( const MultiArrayView< DIM, LABELS>& labels
                          , const MultiArrayView< DIM, WEIGHTS_IN>& weightArray
                          , const ShapeN& roiEnd)
        {
            if (isFinalized_)
            {
                throw std::runtime_error("Segmentor is finalized.  Too late to preprocess.");
            }

            //USETICTOC;

            //TIC;
            graph_.assignLabels(labels, roiEnd);
            //TOC;

            // Use RAG to reshape weights and seeds and resultSegmentation
            resizeArray(edgeWeights_, graph_.edgeNum());
            resizeArray(edgeCounts_, graph_.edgeNum());
            resizeArray<SegmentType>(nodeSeeds_, graph_.nodeNum()+1, EmptySegmentID);
            resizeArray<SegmentType>(resultSegmentation_, graph_.nodeNum()+1, EmptySegmentID);

            //TIC;
            graph_.accumulateEdgeFeatures( labels, weightArray, roiEnd
                                         , edgeWeights_, edgeCounts_);
            //TOC;
        }

        void finalize()
        {
          if (!isFinalized_)
          {
#ifndef NDEBUG
          assert(edgeWeights_.size() == edgeCounts_.size());
#endif
              isFinalized_ = true;

              // Normalize edgeWeights
              #ifdef WITH_OPENMP
              #pragma omp parallel for
              #endif
              for(MultiArrayIndex i=0; i<edgeWeights_.size(); ++i)
              {
                  edgeWeights_[i] /= edgeCounts_[i];
              }
              edgeCounts_.reshape(Shape1(0));
          }

#ifndef NDEBUG
          assert(isFinalized_ && edgeCounts_.size() == 0L);
#endif
        }

        void run(float bias, float noBiasBelow)
        {
            finalize();

            GridSegmentorNodeMap<SegmentType> nodeSeeds(nodeSeeds_);
            GridSegmentorNodeMap<SegmentType> resultSegmentation(resultSegmentation_);
            GridSegmentorEdgeMap<VALUE_TYPE> edgeWeights(edgeWeights_);

            carvingSegmentation( graph_, edgeWeights, nodeSeeds, 1
                               , bias, noBiasBelow, resultSegmentation);
        }

        template<class PIXEL_LABELS>
        void getSegmentation(
              const MultiArrayView<DIM, LABELS>& labels
            , MultiArrayView<DIM, PIXEL_LABELS>& segmentation) const
        {
#ifndef NDEBUG
            validateEqualShapes<DIM>(labels.shape(), segmentation.shape());
#endif
            typedef MultiArrayView<DIM, PIXEL_LABELS> SegView;
            typedef typename SegView::iterator SegIter;

            LabelViewIter labelIter(labels.begin());
            const LabelViewIter labelIterEnd(labels.end());
            SegIter segIter(segmentation.begin());

            for(; labelIter<labelIterEnd; ++labelIter,++segIter)
            {
                const LABELS nodeId = *labelIter;
                *segIter = resultSegmentation_[nodeId];
            }
        }

        void getSuperVoxelSeg(MultiArrayView<1, SegmentType>& segmentation) const
        {
            std::copy( resultSegmentation_.begin(), resultSegmentation_.end()
                     , segmentation.begin());
        }

        void getSuperVoxelSeeds(MultiArrayView<1, SegmentType>& seeds) const
        {
            std::copy(nodeSeeds_.begin(), nodeSeeds_.end()
                     , seeds.begin());
        }

        inline const GridRag<DIM, LABELS>& graph() const
        {
            return graph_;
        }

        inline GridRag<DIM, LABELS>& graph()
        {
            return graph_;
        }

        inline AdjacencyListGraph::index_type nodeNum() const
        {
            return graph_.nodeNum();
        }

        inline AdjacencyListGraph::index_type edgeNum() const
        {
            return graph_.edgeNum();
        }

        inline AdjacencyListGraph::index_type maxNodeId() const
        {
            return graph_.maxNodeId();
        }

        inline AdjacencyListGraph::index_type maxEdgeId() const
        {
            return graph_.maxEdgeId();
        }

        void clearSeeds()
        {
            #ifdef WITH_OPENMP
            #pragma omp parallel for
            #endif
            for(AdjacencyListGraph::index_type i=0; i<nodeNum();++i)
            {
                nodeSeeds_[i] = EmptySegmentID;
            }
        }

        void addSeeds( const MultiArrayView<DIM, LABELS>& labels
                     , const ShapeN& labelsOffset
                     , const MultiArray<2, Int64>& fgSeedsCoord
                     , const MultiArray<2, Int64>& bgSeedsCoord )
        {
            addSeed<BackgroundSegmentID>(labels, labelsOffset, bgSeedsCoord);
            addSeed<ForegroundSegmentID>(labels, labelsOffset, fgSeedsCoord);
        }

        template<class PIXEL_LABELS>
        void addSeedBlock(
              const MultiArrayView<DIM, LABELS>& labels
            , const MultiArrayView<DIM, PIXEL_LABELS>& brushStroke )
        {
#ifndef NDEBUG
            validateEqualShapes<DIM>(labels.shape(), brushStroke.shape());
#endif
            typedef MultiArrayView<DIM, PIXEL_LABELS> BrushView;
            typedef typename BrushView::const_iterator BrushIter;

            LabelViewIter labelIter(labels.begin());
            LabelViewIter labelIterEnd(labels.end());
            BrushIter brushIter(brushStroke.begin());

            for(; labelIter<labelIterEnd; ++labelIter,++brushIter)
            {
                const PIXEL_LABELS brushLabel = *brushIter;
                const LABELS nodeId = *labelIter;

                if(    brushLabel == BackgroundSegmentID
                    || brushLabel == ForegroundSegmentID )
                {
                    nodeSeeds_[nodeId] = brushLabel;
                }
                else if(brushLabel != EmptySegmentID )
                {
                    nodeSeeds_[nodeId] = EmptySegmentID;
                }
            }
        }

        inline const MultiArray<1, VALUE_TYPE>& edgeWeights() const
        {
            return edgeWeights_;
        }

        inline const MultiArray<1, SegmentType>& nodeSeeds() const
        {
            return nodeSeeds_;
        }

        inline const MultiArray<1, SegmentType>& resultSegmentation() const
        {
            return resultSegmentation_;
        }

        inline void clearSegmentation()
        {
            resultSegmentation_ = EmptySegmentID;
        }

        void setResulFgObj(const MultiArray<1, Int64>& fgNodes )
        {
            resultSegmentation_ = BackgroundSegmentID;
            for(MultiArrayIndex i=0; i<fgNodes.shape(0); ++i)
            {
                resultSegmentation_[fgNodes(i)] = ForegroundSegmentID;
            }
        }

        inline bool isFinalized() const
        {
          return isFinalized_;
        }

    private:

        enum SegmentIDs
        {
            EmptySegmentID = 0
          , BackgroundSegmentID = 1
          , ForegroundSegmentID = 2
        };

        template<SegmentType SeedVal>
        void addSeed( const MultiArrayView<DIM, LABELS>& labels
                    , const ShapeN& labelsOffset
                    , const MultiArray<2, Int64>& seedsCoord)
        {
            for(MultiArrayIndex i=0; i<seedsCoord.shape(1); ++i)
            {
                ShapeN c;

                for(std::ptrdiff_t dd=0; dd<DIM; ++dd)
                {
                    // offset coordinates to account of labels offset
                    c[dd] = seedsCoord(dd,i) - labelsOffset[dd];
                }

                if (withinRegion(c, labels.shape()))
                {
                    const MultiArrayIndex node = labels[c];
                    nodeSeeds_[node] = SeedVal;
                }
            }
        }

        /**
         * @brief Evaluates if coordinate position is within region
         * @param coord Coordinate position
         * @param region Region size
         * @return true if coord is with region, false otherwise
         */
        inline bool withinRegion(const ShapeN& coord, const ShapeN& region)
        {
            for(std::ptrdiff_t dd=0; dd<DIM; ++dd)
            {
                if (coord[dd] >= region[dd])
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Grows an array in size, preserving previous values, if possible.
         * @param arr The array to resize
         * @param size The size to grow array to
         * @param init Initial element values for new elements in array
         * @param region Region size
         */
        template<class ElemType>
        void resizeArray(MultiArray<1, ElemType>& arr, size_t size, ElemType init = ElemType(0))
        {
            Shape1 nshape = Shape1(size);

            if (nshape == arr.shape())
            {
              return; // early-out, nothing changes
            }

            MultiArray<1, ElemType> narr(nshape, init);
            narr.subarray(Shape1(0), arr.shape()) = arr;
            arr.swap(narr);
        }



        GridRag<DIM, LABELS> graph_;
        MultiArray<1, VALUE_TYPE> edgeWeights_;
        MultiArray<1, UInt32> edgeCounts_;
        MultiArray<1, SegmentType> nodeSeeds_;
        MultiArray<1, SegmentType> resultSegmentation_;
        bool isFinalized_;
    };
}


#endif /*VIGRA_ILASTIKTOOLS_CARVING_HXX*/
