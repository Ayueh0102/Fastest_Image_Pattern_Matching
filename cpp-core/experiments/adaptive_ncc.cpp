// SPDX-License-Identifier: BSD-2-Clause
#include "adaptive_ncc.hpp"
#include <cfenv>
#include <limits>
#include <utility>

#ifdef __FAST_MATH__
#error Adaptive NCC interval bounds require IEEE arithmetic without fast-math.
#endif

namespace fpm::experimental {
namespace {
constexpr double inf = std::numeric_limits<double>::infinity();
double rectangle(const double* data, std::size_t stride, std::size_t base,
	int x0, int y0, int x1, int y1)
{
	return data[base+std::size_t(y1)*stride+x1]-data[base+std::size_t(y0)*stride+x1]
		-data[base+std::size_t(y1)*stride+x0]+data[base+std::size_t(y0)*stride+x0];
}
}

double AdaptiveNccModel::up(double a) { return std::nextafter(a, inf); }
double AdaptiveNccModel::down(double a) { return std::nextafter(a, -inf); }
AdaptiveNccModel::Interval AdaptiveNccModel::add(Interval a, Interval b)
{
	return {down(a.lo+b.lo), up(a.hi+b.hi)};
}
AdaptiveNccModel::Interval AdaptiveNccModel::subtract(Interval a, Interval b)
{
	return {down(a.lo-b.hi), up(a.hi-b.lo)};
}
AdaptiveNccModel::Interval AdaptiveNccModel::multiplyPositive(Interval a, Interval b)
{
	return {std::max(0.0, down(a.lo*b.lo)), up(a.hi*b.hi)};
}

AdaptiveNccModel::AdaptiveNccModel(const TemplateLevel& templ, int cap, bool blockBound)
	: level_(templ), blockBound_(blockBound)
{
	if (cap != 4 && cap != 8 && cap != 16 && cap != 32 && cap != 64)
		throw std::invalid_argument("adaptive NCC rectangle cap must be 4, 8, 16, 32 or 64");
	detail::validateImage(level_.image.view());
	const int w = level_.image.width(), h = level_.image.height();
	if (!std::numeric_limits<double>::is_iec559 || std::fegetround() != FE_TONEAREST ||
		w < 2 || h < 2 || double(w)*h > 9007199254740992.0/65025) return;
	const std::size_t stride = std::size_t(w)+1;
	std::vector<double> sums(stride*(std::size_t(h)+1)), squares(sums.size());
	buildWorkspaceBytes_ = (sums.capacity()+squares.capacity())*sizeof(double);
	for (int y = 0; y < h; ++y) {
		double s = 0, q = 0;
		for (int x = 0; x < w; ++x) {
			const double v = level_.image.row(y)[x]; s += v; q += v*v;
			const auto i = (std::size_t(y)+1)*stride+x+1;
			sums[i] = sums[i-stride]+s; squares[i] = squares[i-stride]+q;
		}
	}
	nodes_.reserve(2*cap); splits_.reserve(cap-4);
	auto makeNode = [&](int x0, int y0, int x1, int y1) {
		const double n = double(x1-x0)*(y1-y0);
		const double s = rectangle(sums.data(), stride, 0, x0,y0,x1,y1);
		const double q = rectangle(squares.data(), stride, 0, x0,y0,x1,y1);
		const Interval coefficient{std::max(0.0, down(s/n)), up(s/n)};
		const Interval reciprocal{std::max(0.0, down(1/n)), up(1/n)};
		const auto projected = multiplyPositive({s,s}, coefficient);
		nodes_.push_back({x0,y0,x1,y1,n,s,q,coefficient,reciprocal,std::max(0.0,up(q-projected.lo))});
		return int(nodes_.size()-1);
	};
	std::vector<int> active;
	Interval projected;
	for (int by = 0; by < 2; ++by) for (int bx = 0; bx < 2; ++bx) {
		const int id = makeNode(bx*w/2,by*h/2,(bx+1)*w/2,(by+1)*h/2);
		active.push_back(id);
		const auto& n = nodes_[id];
		projected = add(projected, multiplyPositive({n.sum,n.sum},n.coefficient));
	}
	roots_ = rectangles_ = 4;
	templateResidual_[4] = std::max(0.0,up(squares.back()-projected.lo));
	// Greedy template-only splitting. Approximate gain orders work, but never
	// supplies a rejection bound; all actual node bounds are outward rounded.
	struct Choice { double gain = -1; int axis = 0, cut = 0; };
	std::vector<Choice> choices(2*cap);
	auto choose = [&](int id) {
		const auto& n = nodes_[id]; Choice best;
		for (int axis = 0; axis < 2; ++axis) {
			const int start = axis ? n.y0 : n.x0, end = axis ? n.y1 : n.x1;
			for (int cut = start+1; cut < end; ++cut) {
				const double na = axis ? double(n.x1-n.x0)*(cut-n.y0) : double(cut-n.x0)*(n.y1-n.y0);
				const double nb = n.count-na;
				const double sa = rectangle(sums.data(),stride,0,n.x0,n.y0,axis?n.x1:cut,axis?cut:n.y1);
				const double diff = sa/na-(n.sum-sa)/nb;
				const double gain = na*nb/n.count*diff*diff;
				if (gain > best.gain) best = {gain,axis,cut};
			}
		}
		choices[id] = best;
	};
	for (int id : active) choose(id);
	while (rectangles_ < cap) {
		auto where = active.end(); double gain = -1;
		for (auto it = active.begin(); it != active.end(); ++it)
			if (choices[*it].gain > gain) { gain = choices[*it].gain; where = it; }
		if (where == active.end()) break;
		const int parent = *where; const auto n = nodes_[parent]; const auto choice = choices[parent];
		const int left = makeNode(n.x0,n.y0,choice.axis?n.x1:choice.cut,choice.axis?choice.cut:n.y1);
		const int right = makeNode(choice.axis?n.x0:choice.cut,choice.axis?choice.cut:n.y0,n.x1,n.y1);
		active.erase(where); active.push_back(left); active.push_back(right);
		splits_.push_back({parent,left,right}); ++rectangles_;
		projected = subtract(projected, multiplyPositive({n.sum,n.sum},n.coefficient));
		for (int id : {left,right}) {
			choose(id); const auto& child = nodes_[id];
			projected = add(projected,multiplyPositive({child.sum,child.sum},child.coefficient));
		}
		templateResidual_[rectangles_] = std::max(0.0,up(squares.back()-projected.lo));
	}
	buildWorkspaceBytes_ += choices.capacity()*sizeof(Choice)+active.capacity()*sizeof(int);
}

std::size_t AdaptiveNccModel::retainedBytes() const
{
	return sizeof(*this)+nodes_.capacity()*sizeof(Node)+splits_.capacity()*sizeof(Split)+
		std::size_t(level_.image.width())*level_.image.height();
}

bool AdaptiveNccModel::supports(ImageView source) const
{
	return roots_ == 4 && std::fegetround() == FE_TONEAREST &&
		double(source.width)*source.height <= 9007199254740992.0/65025;
}

detail::ScoreMap AdaptiveNccModel::score(ImageView source, double threshold, Backend backend,
	detail::NccStats* stats) const
{
	return detail::nccScreened(source,level_,backend,threshold,*this,stats);
}

bool AdaptiveNccModel::reject(const double* sums, const double* squares, std::size_t stride,
	std::size_t offset, double sum, double square, double denominator, double threshold,
	detail::NccStats* stats) const
{
	if (roots_ != 4 || !std::isfinite(threshold) || threshold <= 0 || threshold > 1 ||
		!std::isfinite(denominator) || denominator <= 0 || std::fegetround() != FE_TONEAREST) return false;
	// Each node's source rectangle sum is an exact integer. All intermediate
	// floating projections enclose their true values; subtract full intervals
	// when replacing a parent, never just subtract an upper bound.
	std::array<Interval,128> products{}, projections{};
	std::array<double,128> residualUpper{};
	Interval product, projection, residual;
	auto insert = [&](int id) {
		const auto& n = nodes_[id];
		const double s = rectangle(sums,stride,offset,n.x0,n.y0,n.x1,n.y1);
		products[id] = multiplyPositive({s,s},n.coefficient);
		projections[id] = multiplyPositive(multiplyPositive({s,s},{s,s}),n.reciprocal);
		product = add(product,products[id]); projection = add(projection,projections[id]);
		if (blockBound_) {
			const double q = rectangle(squares,stride,offset,n.x0,n.y0,n.x1,n.y1);
			const double sourceResidual = std::max(0.0,up(q-projections[id].lo));
			residualUpper[id] = up(std::sqrt(up(sourceResidual*n.residualUpper)));
			// This is the exact FLOAT VALUE of an upper bound, not an interval
			// representing the unknown residual. Its point can safely be removed.
			residual = add(residual,{residualUpper[id],residualUpper[id]});
		}
		if (stats) ++stats->adaptiveRectangles;
	};
	// Covers both separately rounded and fused multiply-subtract evaluation
	// of the legacy numerator P - sum*mean.
	const double centerLower = down(sum*level_.mean);
	auto check = [&](int leaves) {
		if (stats) ++stats->adaptiveChecks;
		double remaining = residual.hi;
		if (!blockBound_) {
			const double sourceResidual = std::max(0.0,up(square-projection.lo));
			remaining = up(std::sqrt(up(sourceResidual*templateResidual_[leaves])));
		}
		const double numerator = up(up(product.hi+remaining)-centerLower);
		const double upperScore = up(numerator/denominator);
		// Positive threshold only: legacy negative saturation can produce zero.
		return std::isfinite(upperScore) && upperScore < threshold;
	};
	for (int i = 0; i < roots_; ++i) insert(i);
	if (check(4)) return true;
	int leaves = 4;
	for (const auto& split : splits_) {
		product = subtract(product,products[split.parent]);
		projection = subtract(projection,projections[split.parent]);
		if (blockBound_) residual = subtract(residual,{residualUpper[split.parent],residualUpper[split.parent]});
		insert(split.left); insert(split.right); ++leaves;
		if (((leaves & (leaves-1)) == 0 || leaves == rectangles_) && check(leaves)) return true;
	}
	return false;
}
}
