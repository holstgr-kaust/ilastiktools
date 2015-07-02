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
    for (int i = 0; i<DIM; ++i)
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
    for (int i = 0; i<DIM; ++i)
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
    public:
        typedef GridGraph<DIM, boost_graph::undirected_tag>  GridGraphType;
        typedef LABELS LabelType;
        typedef TinyVector<MultiArrayIndex, DIM>  ShapeN;
        typedef TinyVector<MultiArrayIndex,   1>  Shape1;

        GridRag() : AdjacencyListGraph()
        {
        }

        /**
         * @brief Find edge id between given labels in graph.
         * @param lu
         * @param lv
         */
        int findEdgeFromIds(const LabelType lu, const LabelType lv)
        {
            const Edge e  = findEdge(nodeFromId(lu), nodeFromId(lv));
            return id(e);
        }

        void assignLabels(
                  const MultiArrayView<DIM, LABELS>& labels
                , const ShapeN roiEnd)
        {
#ifndef NDEBUG
            validateRegionShape(labels.shape(), roiEnd);
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
            validateRegionShape(labels.shape(), roiEnd);
            validateRegionShape(featuresIn.shape(), roiEnd);
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
            // TODO: change size_t -> ptrdiff_t?
            const ShapeN shape = labels.shape();
            for(size_t y=0; y<roiEnd[1]; ++y)
            for(size_t x=0; x<roiEnd[0]; ++x)
            {
                const LabelType l  = labels(x, y);
                if(x+1 < shape[0] )
                    maybeAddEdge(l, labels(x+1, y));
                if(y+1 < shape[1])
                    maybeAddEdge(l, labels(x, y+1));
            }
        }

        // addEdges<3>
        void addEdges( const MultiArrayView<3, LABELS>& labels
                     , const ShapeN& roiEnd)
        {
            // TODO: change size_t -> ptrdiff_t?
            const ShapeN shape = labels.shape();
            for(size_t z=0; z<roiEnd[2]; ++z)
            for(size_t y=0; y<roiEnd[1]; ++y)
            for(size_t x=0; x<roiEnd[0]; ++x)
            {
                const LabelType l  = labels(x, y, z);
                if(x+1 < shape[0] )
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

            // TODO: change size_t -> ptrdiff_t?
            //do the accumulation
            for(size_t y=0; y<roiEnd[1]; ++y)
            for(size_t x=0; x<roiEnd[0]; ++x)
            {
                const LabelType lu  = labels(x, y);

                if(x+1 < shape[0])
                {
                    const LabelType lv = labels(x+1, y);
                    if(lu!=lv)
                    {
                        const int eid = findEdgeFromIds(lu, lv);
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
                        const int eid = findEdgeFromIds(lu, lv);
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
            // TODO: change size_t -> ptrdiff_t?
            #ifdef WITH_OPENMP
            omp_lock_t* edgeLocks = new omp_lock_t[edgeNum()];
            #pragma omp parallel for
            for(size_t i=0; i<edgeNum();++i)
            {
                omp_init_lock(&(edgeLocks[i]));
            }
            #endif

            const ShapeN shape = labels.shape();

            // TODO: change size_t -> ptrdiff_t?
            //do the accumulation
            #ifdef WITH_OPENMP
            #pragma omp parallel for
            #endif
            for(size_t z=0; z<roiEnd[2]; ++z)
            for(size_t y=0; y<roiEnd[1]; ++y)
            for(size_t x=0; x<roiEnd[0]; ++x)
            {
                const LabelType lu  = labels(x, y, z);

                if(x+1 < shape[0])
                {
                    const LabelType lv = labels(x+1, y, z);
                    if(lu!=lv){
                        const int eid = findEdgeFromIds(lu, lv);
                        #ifdef WITH_OPENMP
                        omp_set_lock(&(edgeLocks[eid]));
                        #endif
                        featureCountsOut[eid] += 2;
                        featuresOut[eid] += static_cast<WEIGHTS_OUT>(featuresIn(x,y,z))
                                          + static_cast<WEIGHTS_OUT>(featuresIn(x+1,y,z));
                        #ifdef WITH_OPENMP
                        omp_unset_lock(&(edgeLocks[eid]));
                        #endif
                    }
                }

                if(y+1 < shape[1])
                {
                    const LabelType lv = labels(x, y+1, z);
                    if(lu!=lv)
                    {
                        const int eid = findEdgeFromIds(lu, lv);
                        #ifdef WITH_OPENMP
                        omp_set_lock(&(edgeLocks[eid]));
                        #endif
                        featureCountsOut[eid] += 2;
                        featuresOut[eid] += static_cast<WEIGHTS_OUT>(featuresIn(x,y,z))
                                          + static_cast<WEIGHTS_OUT>(featuresIn(x,y+1,z));
                        #ifdef WITH_OPENMP
                        omp_unset_lock(&(edgeLocks[eid]));
                        #endif
                    }
                }

                if(z+1 < shape[2])
                {
                    const LabelType lv = labels(x, y, z+1);
                    if(lu!=lv)
                    {
                        const int eid = findEdgeFromIds(lu, lv);
                        #ifdef WITH_OPENMP
                        omp_set_lock(&(edgeLocks[eid]));
                        #endif
                        featureCountsOut[eid] += 2;
                        featuresOut[eid] += static_cast<WEIGHTS_OUT>(featuresIn(x,y,z))
                                          + static_cast<WEIGHTS_OUT>(featuresIn(x,y,z+1));
                        #ifdef WITH_OPENMP
                        omp_unset_lock(&(edgeLocks[eid]));
                        #endif
                    }
                }
            }

            // TODO: change size_t -> ptrdiff_t?
            #ifdef WITH_OPENMP
            #pragma omp parallel for
            for(size_t i=0; i<edgeNum();++i)
            {
                omp_destroy_lock(&(edgeLocks[i]));
            }
            delete[] edgeLocks;
            #endif
        }

        /**
         * @brief add label edges to graph is they are different.
         * @param lu
         * @param lv
         */
        void maybeAddEdge(const LabelType lu, const LabelType lv)
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
            for (LABELS id = maxNodeId() + 1; id < maxLabel; ++id)
            {
                addNode(id);
            }
        }
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
        Reference operator[](const K key)
        {
            return values_[key.id()];
        }

        template<class K>
        ConstReference operator[](const K key)const
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
        Reference operator[](const K key)
        {
            return values_[key.id()];
        }

        template<class K>
        ConstReference operator[](const K key)const
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

            // Get the RAG
            //std::cout<<"get RAG\n";
            //TIC;
            graph_.assignLabels(labels, roiEnd);
            //TOC;

            // Use RAG to reshape weights and seeds and resultSegmentation
            edgeWeights_.reshape(Shape1(graph_.edgeNum()));
            edgeCounts_.reshape(Shape1(graph_.edgeNum()));
            nodeSeeds_.reshape(Shape1(graph_.maxNodeId()+ 1));
            resultSegmentation_.reshape(Shape1(graph_.maxNodeId()+ 1));

            // Accumulate the edge weights
            //std::cout<<"get edge weights\n";
            //TIC;
            graph_.accumulateEdgeFeatures( labels, weightArray, roiEnd
                                         , edgeWeights_, edgeCounts_);
            //TOC;
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

            //std::cout<<"get RAG from serialization\n";
            //TIC;
            // get the RAG
            graph_.assignLabelsFromSerialization(serialization);
            //TOC;

            // Assign weights and seeds and resultSegmentation
            edgeWeights_ = edgeWeights;
            edgeCounts_.reshape(edgeWeights.shape());
            edgeCounts_ = 1;
            nodeSeeds_ = nodeSeeds;
            resultSegmentation_ = resultSegmentation;

            isFinalized_ = true;
        }

        void run(float bias, float noBiasBelow)
        {
#ifndef NDEBUG
            assert(edgeWeights_.size() == edgeCounts_.size());
#endif
            if (!isFinalized_)
            {
                isFinalized_ = true;

                // TODO: change size_t -> ptrdiff_t?
                // Normalize edgeWeights
                #ifdef WITH_OPENMP
                #pragma omp parallel for
                #endif
                for(size_t i=0; i<edgeWeights_.size(); ++i)
                {
                    edgeWeights_[i] /= edgeCounts_[i];
                    edgeCounts_[i] = 1;
                }
            }

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
            validateEqualShapes(labels.shape(), segmentation.shape());
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

        const GridRag<DIM, LABELS>& graph() const
        {
            return graph_;
        }

        GridRag<DIM, LABELS>& graph()
        {
            return graph_;
        }

        size_t nodeNum() const
        {
            return graph_.nodeNum();
        }

        size_t edgeNum() const
        {
            return graph_.edgeNum();
        }

        size_t maxNodeId() const
        {
            return graph_.maxNodeId();
        }

        size_t maxEdgeId() const
        {
            return graph_.maxEdgeId();
        }

        void clearSeeds()
        {
            // TODO: change size_t -> ptrdiff_t?
            #ifdef WITH_OPENMP
            #pragma omp parallel for
            #endif
            for(size_t i=0; i<nodeNum();++i)
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
            validateEqualShapes(labels.shape(), brushStroke.shape());
#endif
            typedef MultiArrayView<DIM, PIXEL_LABELS> BrushView;
            typedef typename BrushView::const_iterator BrushIter;

            LabelViewIter labelIter(labels.begin());
            LabelViewIter labelIterEnd(labels.end());
            BrushIter brushIter(brushStroke.begin());

            for(; labelIter<labelIterEnd; ++labelIter,++brushIter){
                const int brushLabel = int(*brushIter);
                const LABELS nodeId = *labelIter;

                //std::cout<<"brush label "<<brushLabel<<"\n";
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

        const MultiArray<1 , VALUE_TYPE>& edgeWeights() const
        {
            return edgeWeights_;
        }

        const MultiArray<1 , SegmentType>& nodeSeeds() const
        {
            return nodeSeeds_;
        }

        const MultiArray<1 , SegmentType>& resultSegmentation() const
        {
            return resultSegmentation_;
        }

        void clearSegmentation()
        {
            resultSegmentation_ = EmptySegmentID;
        }

        void setResulFgObj(const MultiArray<1, Int64>& fgNodes )
        {
            resultSegmentation_ = BackgroundSegmentID;
            // TODO: change size_t -> ptrdiff_t?
            for(size_t i=0; i<fgNodes.shape(0); ++i)
            {
                resultSegmentation_[fgNodes(i)] = ForegroundSegmentID;
            }
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
            // TODO: change size_t -> ptrdiff_t?
            for(size_t i=0; i<seedsCoord.shape(1); ++i)
            {
                ShapeN c;

                // TODO: change int -> size_t or ptrdiff_t?
                for(int dd=0; dd<DIM; ++dd)
                {
                    // offset coordinates to account of labels offset
                    c[dd] = seedsCoord(dd,i) - labelsOffset[dd];
                }

                if (withinRegion(c, labels.shape()))
                {
                    const UInt64 node = labels[c];
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
        bool withinRegion(const ShapeN& coord, const ShapeN& region)
        {
            // TODO: change int -> size_t or ptrdiff_t?
            for(int dd=0; dd<DIM; ++dd)
            {
                if (coord[dd] >= region[dd]) return false;
            }
            return true;
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
