/************************************************************************/
/*                                                                      */
/*                 Copyright 2011 by Ullrich Koethe                     */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#define PY_ARRAY_UNIQUE_SYMBOL vigranumpyilastiktools_PyArray_API
//#define NO_IMPORT_ARRAY

#include <utility>
#include <vector>
#include <unordered_set>
#include <unordered_map>

// Include this first to avoid name conflicts for boost::tie,
// similar to issue described in vigra#237
#include <boost/tuple/tuple.hpp>
#include <boost/functional/hash.hpp>

// vigra
#include <ilastiktools/carving.hxx>

// vigra python
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

// #include "export_graph_visitor.hxx"
// #include "export_graph_rag_visitor.hxx"
// #include "export_graph_algorithm_visitor.hxx"
// #include "export_graph_shortest_path_visitor.hxx"
// #include "export_graph_hierarchical_clustering_visitor.hxx"

namespace python = boost::python;

using namespace vigra;
using namespace boost::python;


template<unsigned int DIM, class LABEL_TYPE>
void pyAssignLabels( GridRag<DIM, LABEL_TYPE>& graph
                   , const NumpyArray<DIM, LABEL_TYPE>& labels
                   , const TinyVector<MultiArrayIndex, DIM>& roiEnd )
{
    graph.assignLabels(labels, roiEnd);
}

template<unsigned int DIM, class LABEL_TYPE>
void pyAssignLabelsFromSerialization(
      GridRag<DIM, LABEL_TYPE>& graph
    , const NumpyArray<1, UInt32>& serialization )
{
    graph.assignLabelsFromSerialization(serialization);
}

template<unsigned int DIM, class LABEL_TYPE, class FEATURES_IN>
NumpyAnyArray pyAccumulateEdgeFeatures(
    GridRag<DIM, LABEL_TYPE>& graph
  , const NumpyArray<DIM, LABEL_TYPE>& labels
  , const NumpyArray<DIM, FEATURES_IN>& featuresIn
  , const TinyVector<MultiArrayIndex, DIM>& roiEnd
  , NumpyArray<1, typename NumericTraits<FEATURES_IN>::RealPromote> featuresOut
  , NumpyArray<1, UInt32>& featureCountsOut )
{
    typedef TinyVector<MultiArrayIndex, 1>  Shape1;
    Shape1 shape(graph.edgeNum());
    featuresOut.reshapeIfEmpty(shape);
    featureCountsOut.reshapeIfEmpty(shape);

    graph.accumulateEdgeFeatures( labels, featuresIn, roiEnd
                                , featuresOut, featureCountsOut );
    return featuresOut;
}

template<unsigned int DIM, class LABEL_TYPE>
void pyPreprocessing(
    GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , const NumpyArray<DIM, LABEL_TYPE>& labels
  , const NumpyArray<DIM, float>& weightArray
  , const TinyVector<MultiArrayIndex, DIM>& roiEnd )
{
    gridSegmentor.preprocessing(labels, weightArray, roiEnd);
}

template<unsigned int DIM, class LABEL_TYPE>
void pyInitFromSerialization(
    GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , const NumpyArray<1, LABEL_TYPE>& serialization
  , const NumpyArray<1, float>& edgeWeights
  , const NumpyArray<1, UInt8>& nodeSeeds
  , const NumpyArray<1, UInt8>& resultSegmentation )
{
    gridSegmentor.initFromSerialization(serialization,
                                        edgeWeights, nodeSeeds,
                                        resultSegmentation);
}

template<unsigned int DIM, class LABEL_TYPE>
void pyAddSeeds(
    GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , const NumpyArray<DIM, LABEL_TYPE>& labels
  , const TinyVector<MultiArrayIndex, DIM>& labelsOffset
  , const NumpyArray<2, Int64>& fgSeeds
  , const NumpyArray<2, Int64>& bgSeeds )
{
    gridSegmentor.addSeeds(labels, labelsOffset, fgSeeds, bgSeeds);
}

template<unsigned int DIM, class LABEL_TYPE>
void pyAddSeedBlock(
    GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , const NumpyArray<DIM, LABEL_TYPE>& labels
  , const NumpyArray<DIM, UInt8>& brushStroke )
{
    gridSegmentor.addSeedBlock(labels, brushStroke);
}

template<unsigned int DIM, class LABEL_TYPE>
NumpyAnyArray pyGetSegmentation(
    const GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , const NumpyArray<DIM, LABEL_TYPE>& labels
  , NumpyArray<DIM, UInt8>  segmentation )
{
    typedef TinyVector<MultiArrayIndex, DIM>  ShapeN;
    ShapeN shape(labels.shape());
    segmentation.reshapeIfEmpty(shape);

    {
        PyAllowThreads _pythread;
        gridSegmentor.getSegmentation(labels, segmentation);
    }
   
    return segmentation;
}


template<unsigned int DIM, class LABEL_TYPE>
NumpyAnyArray pyGetSuperVoxelSeg(
    const GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , NumpyArray<1, UInt8>  segmentation )
{
    typedef TinyVector<MultiArrayIndex, 1>  Shape1;
    Shape1 shape(gridSegmentor.maxNodeId()+1);
    segmentation.reshapeIfEmpty(shape);

    {
        PyAllowThreads _pythread;
        gridSegmentor.getSuperVoxelSeg(segmentation);
    }
   
    return segmentation;
}

template<unsigned int DIM, class LABEL_TYPE>
NumpyAnyArray pyGetSuperVoxelSeeds(
    const GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , NumpyArray<1, UInt8>  seeds )
{
    typedef TinyVector<MultiArrayIndex, 1>  Shape1;
    Shape1 shape(gridSegmentor.maxNodeId()+1);
    seeds.reshapeIfEmpty(shape);

    {
        PyAllowThreads _pythread;
        gridSegmentor.getSuperVoxelSeeds(seeds);
    }
   
    return seeds;
}


template<unsigned int DIM, class LABEL_TYPE>
NumpyAnyArray pySerializeGraph(
    const GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , NumpyArray<1, UInt32> serialization )
{
    typedef NumpyArray<1, UInt32>::difference_type DifferenceType;

    serialization.reshapeIfEmpty(DifferenceType(gridSegmentor.graph().serializationSize()));
    gridSegmentor.graph().serialize(serialization.begin());
    return serialization;
}

template<unsigned int DIM, class LABEL_TYPE>
NumpyAnyArray pyEdgeWeights(
    const GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , NumpyArray<1, float> out )
{
    typedef NumpyArray<1, UInt32>::difference_type DifferenceType;

    assert(gridSegmentor.isFinalized()); // weights were not scaled correctly

    out.reshapeIfEmpty(DifferenceType(gridSegmentor.edgeNum()));
    out = gridSegmentor.edgeWeights();
    return out;
}

template<unsigned int DIM, class LABEL_TYPE>
NumpyAnyArray pyNodeSeeds(
    const GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , NumpyArray<1, UInt8> out )
{
    out.reshapeIfEmpty(gridSegmentor.nodeSeeds().shape());
    out = gridSegmentor.nodeSeeds();
    return out;
}

template<unsigned int DIM, class LABEL_TYPE>
NumpyAnyArray pyGetResultSegmentation(
    const GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , NumpyArray<1, UInt8> out )
{
    out.reshapeIfEmpty(gridSegmentor.resultSegmentation().shape());
    out = gridSegmentor.resultSegmentation();
    return out;
}

template<unsigned int DIM, class LABEL_TYPE>
void pySetResulFgObj(
    GridSegmentor<DIM, LABEL_TYPE, float>& gridSegmentor
  , const NumpyArray<1, Int64>& fgNodes )
{
    gridSegmentor.setResulFgObj(fgNodes);
}

template<unsigned int DIM, class LABEL_TYPE>
void defineGridRag(const std::string & clsName){
    typedef GridRag<DIM, LABEL_TYPE> Graph;
    python::class_<Graph>(clsName.c_str(),python::init<  >())
        .def("assignLabels",
            registerConverters(&pyAssignLabels<DIM, LABEL_TYPE>),
            (
                python::arg("labels")
              , python::arg("roiEnd")
            )
        )
        .def("accumulateEdgeFeatures", 
            registerConverters(&pyAccumulateEdgeFeatures<DIM, LABEL_TYPE, float>),
            (
                python::arg("labels")
              , python::arg("featuresIn")
              , python::arg("roiEnd")
              , python::arg("featuresOut") = python::object()
              , python::arg("featureCountsOut") = python::object()
            )
        )
    ;
}

template<unsigned int DIM, class LABEL_TYPE>
void defineGridSegmentor(const std::string & clsName)
{
    typedef GridSegmentor<DIM, LABEL_TYPE, float> Segmentor;

    python::class_<Segmentor>(clsName.c_str(),python::init<  >())
        .def("preprocessing", 
            registerConverters( & pyPreprocessing<DIM, LABEL_TYPE>),
            (
                python::arg("labels")
              , python::arg("weightArray")
              , python::arg("roiEnd")
            )
        )
        .def("init", &Segmentor::init)
        .def("initFromSerialization",
            registerConverters( & pyInitFromSerialization<DIM, LABEL_TYPE>),
            (
                python::arg("serialization")
              , python::arg("edgeWeights")
              , python::arg("nodeSeeds")
              , python::arg("resultSegmentation")
            )
        )

        .def("clearSeeds",&Segmentor::clearSeeds)
        .def("addSeedBlock",
            registerConverters( & pyAddSeedBlock<DIM, LABEL_TYPE>),
            (
                python::arg("labels")
              , python::arg("brushStroke")
            )
        )
        .def("addSeeds",
            registerConverters( & pyAddSeeds<DIM, LABEL_TYPE>),
            (
                 python::arg("labels")
               , python::arg("labelsOffset")
               , python::arg("fgSeeds")
               , python::arg("bgSeeds")
            )
        )
        .def("clearSegmentation",&Segmentor::clearSegmentation)
        .def("setResulFgObj", 
            registerConverters( & pySetResulFgObj<DIM, LABEL_TYPE>),
            (
                python::arg("fgNodes")
            )
        )
        .def("getSegmentation", 
            registerConverters( & pyGetSegmentation<DIM, LABEL_TYPE>),
            (
                python::arg("labels")
              , python::arg("out") = python::object()
            )
        )
        .def("nodeNum",&Segmentor::nodeNum)
        .def("edgeNum",&Segmentor::edgeNum)
        .def("maxNodeId",&Segmentor::maxNodeId)
        .def("maxEdgeId",&Segmentor::maxEdgeId)
        .def("run",&Segmentor::run)
        .def("finalize",&Segmentor::finalize)
        .def("isFinalized",&Segmentor::isFinalized)
        .def("serializeGraph", registerConverters(&pySerializeGraph<DIM, LABEL_TYPE>),
            (
                python::arg("out") = python::object()
            )
        )
        .def("getEdgeWeights",registerConverters(pyEdgeWeights<DIM, LABEL_TYPE>),
            (
                python::arg("out") = python::object()
            )
        )
        .def("getNodeSeeds",registerConverters(pyNodeSeeds<DIM, LABEL_TYPE>),
            (
                python::arg("out") = python::object()
            )
        )
        .def("getResultSegmentation",registerConverters(pyGetResultSegmentation<DIM, LABEL_TYPE>),
            (
                python::arg("out") = python::object()
            )
        )
        .def("getSuperVoxelSeg",registerConverters(pyGetSuperVoxelSeg<DIM, LABEL_TYPE>),
            (
                python::arg("out") = python::object()
            )
        )
        .def("getSuperVoxelSeeds",registerConverters(pyGetSuperVoxelSeeds<DIM, LABEL_TYPE>),
            (
                python::arg("out") = python::object()
            )
        )
    ;
}

// Define lookup type:
// (u,v) -> [(x,y), (x,y),...]
typedef std::pair<UInt32, UInt32> edge_id_t;
typedef std::unordered_map<edge_id_t,
                           std::vector<Shape2>,
                           boost::hash<edge_id_t> > edge_coord_lookup_t;

typedef std::pair<edge_coord_lookup_t, edge_coord_lookup_t> edge_coord_lookup_pair_t;

//*****************************************************************************
//* Given a 2D label image, find all the coordinates just *before* a label
//* transition in both the x and y directions.
//*
//* Returns a pair of mappings of edge pairs (u,v) to coordinate lists; one for
//* edges in the horizontal direction, and another for the vertical direction.
//*
//* For every edge pair (u,v): u < v.
//* No ordering guarantee is made for the coordinate lists (but in practice,
//* they will be in scan order).
//*
//* As a convenience, every key in the horizontal lookup is guaranteed to exist
//* in the vertical lookup (and vice-versa), even if that key's coordinate list
//* is empty.
//*
//*****************************************************************************
edge_coord_lookup_pair_t edgeCoords2D( MultiArrayView<2, UInt32> const & src )
{
    using namespace vigra;
    MultiArrayIndex x_dim = src.shape(0);
    MultiArrayIndex y_dim = src.shape(1);

    edge_coord_lookup_t horizontal_edge_coords;
    edge_coord_lookup_t vertical_edge_coords;

    for (MultiArrayIndex y = 0; y < y_dim; ++y)
    {
        for (MultiArrayIndex x = 0; x < x_dim; ++x)
        {
            // Lambda to append to a lookup
            auto append_to_lookup =
            [&](edge_coord_lookup_t & lookup, MultiArrayIndex x1, MultiArrayIndex y1)
            {
                auto u = src(x,y);
                auto v = src(x1,y1);
                edge_id_t edge_id = (u < v) ? std::make_pair(u,v) : std::make_pair(v,u);

                auto iter = lookup.find(edge_id);
                if ( iter == lookup.end() )
                {
                    // Edge not yet seen. Create a new coord vector
                    auto coord_list = std::vector<Shape2>();
                    coord_list.push_back(Shape2(x,y));
                    lookup[edge_id] = coord_list;
                }
                else
                {
                    // Append to coord vector
                    auto & coord_list = iter->second;
                    coord_list.push_back(Shape2(x,y));
                }
            };

            // Check to the right
            if (x < x_dim-1 && src(x,y) != src(x+1,y))
            {
                append_to_lookup(horizontal_edge_coords, x+1, y);
            }

            // Check below
            if (y < y_dim-1 && src(x,y) != src(x,y+1))
            {
                append_to_lookup(vertical_edge_coords, x, y+1);
            }
        }
    }

    // Convenience feature:
    // Ensure that every pair (u,v) in horizontal_edge_coords appears in vertical_edge_coords
    // and vice-versa, even if it maps to an empty coordinate list in one of them.
    auto fill_missing_keys =
    [](edge_coord_lookup_t const & from, edge_coord_lookup_t & to)
    {
        for ( auto const & k_v : from )
        {
            if ( to.find(k_v.first) == to.end() )
            {
                to[k_v.first] = std::vector<Shape2>();
            }
        }
    };
    fill_missing_keys(horizontal_edge_coords, vertical_edge_coords);
    fill_missing_keys(vertical_edge_coords, horizontal_edge_coords);

    return std::make_pair(horizontal_edge_coords, vertical_edge_coords);
}

//*****************************************************************************
//* Convert an edge_coord_lookup_t to a corresponding python structure.
//* Result is a dict of { tuple : list-of-tuple }, like this:
//*   { (u,v) : [(x,y), (x,y), (x,y), ...] }
//*****************************************************************************
python::dict edgeCoordLookupToPython( edge_coord_lookup_t const & edge_coord_lookup )
{
    namespace py = boost::python;

    py::dict pylookup;
    for ( auto & edge_and_coords : edge_coord_lookup )
    {
        edge_id_t const & edge_id = edge_and_coords.first;
        std::vector<Shape2> const & coords = edge_and_coords.second;

        py::list pycoords;
        for ( auto coord : coords )
        {
            pycoords.append(py::make_tuple(coord[0], coord[1]));
        }
        pylookup[py::make_tuple(edge_id.first, edge_id.second)] = pycoords;
    }
    return pylookup;
}

//*****************************************************************************
//* Python function for edgeCoords2D().
//* See edgeCoords2D() documentation for details.
//*****************************************************************************
python::tuple pythonEdgeCoords2D(NumpyArray<2, UInt32> const & src)
{
    edge_coord_lookup_pair_t lookup_pair;
    {
        PyAllowThreads _pythread;
        lookup_pair = edgeCoords2D(src); // C++ move constructor should work here ...right?
    }
    namespace py = boost::python;
    py::dict pycoords_horizontal = edgeCoordLookupToPython(lookup_pair.first);
    py::dict pycoords_vertical = edgeCoordLookupToPython(lookup_pair.second);
    return py::make_tuple( pycoords_horizontal, pycoords_vertical );
}


//*****************************************************************************
//* Find all the label transitions for the given 2D label image.
//* Then return a dict-of-lists mapping each edge (u,v) to a list of line
//* segments that could be drawn on screen to represent the edge.
//*
//* Conceptually, the output looks like this:
//*
//* { (u,v) : [ ((x1,y1), (x2,y2)),
//*             ((x1,y1), (x2,y2)),
//*             ((x1,y1), (x2,y2)),
//*             ... ] }
//*
//* ...but the line segment list is returned as a NumpyArray of shape (N,2,2).
//*
//* Note: The line segments are not guaranteed to appear in any particular order.
//*
//* TODO: It would be nice if we could use PyAllowThreads here, but I'm not
//*       sure if it's allowed -- we create NumpyArrays inside the loop
//*       (of type line_segment_array_t) inside the loop.
//*
//*****************************************************************************
python::dict line_segments_for_labels( NumpyArray<2, UInt32> label_img )
{
    typedef NumpyArray<3, UInt32> line_segment_array_t;
    typedef std::unordered_map<edge_id_t, line_segment_array_t, boost::hash<edge_id_t> > line_segment_lookup_t;

    auto lookup_pair = edgeCoords2D(label_img);
    auto const & horizontal_coord_lookup = lookup_pair.first;
    auto const & vertical_coord_lookup = lookup_pair.second;

    typedef std::unordered_set<edge_id_t, boost::hash<edge_id_t> > edge_id_set_t;
    edge_id_set_t all_edge_ids;
    for ( auto const & k_v : horizontal_coord_lookup )
    {
        all_edge_ids.insert(k_v.first);
    }
    for ( auto const & k_v : vertical_coord_lookup )
    {
        all_edge_ids.insert(k_v.first);
    }

    line_segment_lookup_t line_seg_lookup;
    for ( auto const & edge_id : all_edge_ids )
    {
        // Empty by default
        std::vector<Shape2> horizontal_edge_coords;
        std::vector<Shape2> vertical_edge_coords;

        // Overwrite if found
        auto iter_horizontal_coords = horizontal_coord_lookup.find(edge_id);
        if ( iter_horizontal_coords != horizontal_coord_lookup.end() )
        {
            horizontal_edge_coords = iter_horizontal_coords->second;
        }

        // Overwrite if found
        auto iter_vertical_coords = vertical_coord_lookup.find(edge_id);
        if ( iter_vertical_coords != vertical_coord_lookup.end() )
        {
            vertical_edge_coords = iter_vertical_coords->second;
        }

        auto num_segments = horizontal_edge_coords.size() + vertical_edge_coords.size();
        line_seg_lookup[edge_id] = line_segment_array_t(Shape3(num_segments, 2, 2));
        auto line_segments = line_seg_lookup[edge_id];

        // Line segments to the RIGHT of the HORIZONTAL edge coordinates
        for ( int i = 0; i < horizontal_edge_coords.size(); ++i )
        {
            line_segments(i, 0, 0) = horizontal_edge_coords[i][0] + 1;
            line_segments(i, 0, 1) = horizontal_edge_coords[i][1] + 0;
            line_segments(i, 1, 0) = horizontal_edge_coords[i][0] + 1;
            line_segments(i, 1, 1) = horizontal_edge_coords[i][1] + 1;
        }
        // Line segments BELOW the VERTICAL edge coordinates
        auto offset = horizontal_edge_coords.size();
        for ( int i = 0; i < vertical_edge_coords.size(); ++i )
        {
            line_segments(offset+i, 0, 0) = vertical_edge_coords[i][0] + 0;
            line_segments(offset+i, 0, 1) = vertical_edge_coords[i][1] + 1;
            line_segments(offset+i, 1, 0) = vertical_edge_coords[i][0] + 1;
            line_segments(offset+i, 1, 1) = vertical_edge_coords[i][1] + 1;
        }
    }

    namespace py = boost::python;
    py::dict ret;
    for ( auto const & k_v : line_seg_lookup )
    {
        auto const & edge_id = k_v.first;
        ret[py::make_tuple(edge_id.first, edge_id.second)] = k_v.second;
    }
    return ret;
}

BOOST_PYTHON_MODULE_INIT(_core)
{
    import_vigranumpy();

    python::docstring_options doc_options(true, true, false);

    defineGridRag<2, vigra::UInt32>("GridRag_2D_UInt32");
    defineGridSegmentor<2, vigra::UInt32>("GridSegmentor_2D_UInt32");

    defineGridRag<3, vigra::UInt32>("GridRag_3D_UInt32");
    defineGridSegmentor<3, vigra::UInt32>("GridSegmentor_3D_UInt32");

    using namespace boost::python;
    def("edgeCoords2D", registerConverters(&pythonEdgeCoords2D), (arg("src")));
    def("line_segments_for_labels", registerConverters(&line_segments_for_labels), (arg("label_img")));
}


