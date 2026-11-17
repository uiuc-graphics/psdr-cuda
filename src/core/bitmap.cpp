#include <misc/Exception.h>
#include <psdr/core/bitmap_loader.h>
#include <psdr/core/bitmap.h>
#include <psdr/core/transform.h>
#include <drjit/math.h>
#include <drjit/idiv.h>

NAMESPACE_BEGIN(psdr_jit)

template <int channels>
Bitmap<channels>::Bitmap()
    : m_resolution(1, 1), m_data(zeros<ValueD>()) {drjit::make_opaque(m_scale, m_rot, m_trans, m_data);}


template <int channels>
Bitmap<channels>::Bitmap(const char *file_name) {
    load_openexr(file_name);
    drjit::make_opaque(m_scale, m_rot, m_trans, m_data);
}


template <int channels>
Bitmap<channels>::Bitmap(ScalarValue value) : m_resolution(1, 1), m_data(value) {drjit::make_opaque(m_scale, m_rot, m_trans, m_data);}


template <int channels>
Bitmap<channels>::Bitmap(int width, int height, const ValueD &data) : m_resolution(width, height), m_data(data) {
    // TODO: fix this
    // PSDR_ASSERT(width*height*channels == static_cast<int>(data.size()));
    drjit::make_opaque(m_scale, m_rot, m_trans, m_data);
}


template <int channels>
void Bitmap<channels>::load_openexr(const char *file_name) {
    Vector4fC data;
    std::tie(data, m_resolution) = BitmapLoader::load_openexr_rgba(file_name);
    if constexpr ( channels == 1 ) {
        m_data = data[0];
    } else {
        m_data = Vector3fD(data[0], data[1], data[2]);
    }
}


template <int channels>
template <bool ad>
typename Bitmap<channels>::template Value<ad> Bitmap<channels>::eval(Vector2f<ad> uv, bool flip_v, bool envmap_mode) const {
    const int width = m_resolution.x(), height = m_resolution.y();

    if ( static_cast<int>(slices<ValueD>(m_data)) != width*height )
        throw Exception("Bitmap: invalid data size!");

    if ( width == 1 && height == 1 ) {
        if constexpr ( ad )
            return m_data;
        else
            return detach<false>(m_data);
    } else {
        if ( width < 2 || height < 2 )
            throw Exception("Bitmap: invalid resolution!");

        if constexpr ( ad ) {
            uv = Vector2fD((uv.x()-0.5f)*cos((m_rot)) + (uv.y()-0.5f)*sin((m_rot)), -(uv.x()-0.5f)*sin((m_rot)) + (uv.y()-0.5f)*cos((m_rot)));
            uv += .5f;
            if ( flip_v ) {
                // flip the v coordinates to match common practices
                uv.y() = -uv.y();
            }
            uv *= (m_scale);
            uv.x() -= -.5f+(m_scale)/2;
            uv.y() += -.5f+(m_scale)/2;
            uv += (m_trans);
        } else {
            uv = Vector2fC((uv.x()-0.5f)*cos(detach<false>(m_rot)) + (uv.y()-0.5f)*sin(detach<false>(m_rot)), -(uv.x()-0.5f)*sin(detach<false>(m_rot)) + (uv.y()-0.5f)*cos(detach<false>(m_rot)));
            uv += .5f;
            if ( flip_v ) {
                // flip the v coordinates to match common practices
                uv.y() = -uv.y();
            }
            uv *= detach<false>(m_scale);
            uv.x() -= -.5f+detach<false>(m_scale)/2;
            uv.y() += -.5f+detach<false>(m_scale)/2;
            uv += detach<false>(m_trans);
        }

        if (envmap_mode) {
            uv.x() -= 0.5 / width;
            uv -= floor(uv);
            uv.x() *= m_resolution.x();
            uv.y() *= m_resolution.y() - 1;
        }
        else {
            uv -= floor(uv);
            uv *= m_resolution - 1;
        }

        Vector2i<ad> pos = floor2int<Vector2i<ad>, Vector2f<ad>>(uv);
        Vector2f<ad> w1 = uv - Vector2f<ad>(pos), w0 = 1.0f - w1;

        Int<ad> yw, xp1;
        if (envmap_mode) {
            yw = minimum(pos.y(), m_resolution.y() - 2) * width;
            xp1 = imod(pos.x() + 1, m_resolution.x()); 
        }
        else {
            pos = minimum(pos, m_resolution - 2);
            yw = pos.y() * width;
            xp1 = pos.x() + 1; 
        }

        Value<ad> v00, v10, v01, v11;
        if constexpr ( ad ) {
            v00 = gather<ValueD>(m_data, yw + pos.x());
            v10 = gather<ValueD>(m_data, yw + xp1);
            v01 = gather<ValueD>(m_data, yw + pos.x() + width);
            v11 = gather<ValueD>(m_data, yw + xp1 + width);
        } else {
            const ValueC &data = detach<false>(m_data);
            v00 = gather<ValueC>(data, yw + pos.x());
            v10 = gather<ValueC>(data, yw + xp1);
            v01 = gather<ValueC>(data, yw + pos.x() + width);
            v11 = gather<ValueC>(data, yw + xp1 + width);
        }

        // Bilinear interpolation
        Value<ad> v0 = fmadd(w0.x(), v00, w1.x()*v10),
                  v1 = fmadd(w0.x(), v01, w1.x()*v11);
        return fmadd(w0.y(), v0, w1.y()*v1);
    }
}


// Explicit instantiations
template struct Bitmap<1>;
template struct Bitmap<3>;

template FloatC Bitmap<1>::eval<false>(Vector2fC, bool, bool) const;
template FloatD Bitmap<1>::eval<true>(Vector2fD, bool, bool) const;

template Vector3fC Bitmap<3>::eval<false>(Vector2fC, bool, bool) const;
template Vector3fD Bitmap<3>::eval<true>(Vector2fD, bool, bool) const;

NAMESPACE_END(psdr_jit)
